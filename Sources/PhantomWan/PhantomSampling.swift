import Foundation
import MLX
import MLXRandom
import WanCore

/// Phantom subject-to-video sampler — THREE DiT forwards per step (verbatim oracle `sample_s2v`):
///   pos_it = model(cat[target, refs],      text)   — text + subject
///   pos_i  = model(cat[target, refs],      null)   — subject only
///   neg    = model(cat[target, ZERO refs], null)   — neither (zeroed-ref "uncond")
///   noise_pred = neg + w_img·(pos_i − neg) + w_text·(pos_it − pos_i)
/// The clean refs are re-clamped into the latent tail every step (refs never noised); the K ref
/// frames are stripped after the loop. Stock wan-core `WanModel` (no control branch) + FlowUniPC.
public func sampleS2V(
    model: WanModel,
    refLatents: MLXArray,       // [16, K, h, w] clean reference latents
    contextCond: MLXArray,      // raw umT5 features [L, text_dim] (positive)
    contextNull: MLXArray,      // raw umT5 features [L, text_dim] (null/neg)
    config: WanConfig,
    fLatent: Int,               // F target latent frames
    hLatent: Int,
    wLatent: Int,
    steps: Int,
    shift: Double,
    guideImg: Double = 5.0,
    guideText: Double = 7.5,
    seed: UInt64? = nil,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
) rethrows -> MLXArray {
    let k = refLatents.dim(1)
    let refsNeg = MLXArray.zeros(like: refLatents)

    // Extended (F+K) grid — the refs rope at sequential positions F..F+K-1 (stock forward).
    let ps = config.patchSize
    let grid = ((fLatent + k) / ps[0], hLatent / ps[1], wLatent / ps[2])
    let seqLen = grid.0 * grid.1 * grid.2
    let rope = model.prepareRope([grid])

    let sched = FlowUniPCScheduler(numTrainTimesteps: config.numTrainTimesteps)
    sched.setTimesteps(steps, shift: shift)
    let timesteps = sched.timesteps

    if let seed { MLXRandom.seed(seed) }
    var latent = MLXRandom.normal([config.vaeZDim, fLatent + k, hLatent, wLatent])  // [16, F+K, h, w]

    // Cap the buffer cache during the denoise (the wan-family lever; 3 forwards/step is heavy).
    let prevCacheLimit = Memory.cacheLimit
    let capMB = ProcessInfo.processInfo.environment["DENOISE_CACHE_MB"].flatMap { Int($0) } ?? 2048
    Memory.cacheLimit = capMB * 1_000_000
    defer { Memory.cacheLimit = prevCacheLimit }

    for i in 0..<steps {
        let t = Float(timesteps[i])
        let tArr = MLXArray([t])
        let noisyTarget = latent[0..., 0 ..< fLatent]                   // [16, F, h, w]
        let inpRefs = concatenated([noisyTarget, refLatents], axis: 1)  // re-clamp clean refs
        let inpZero = concatenated([noisyTarget, refsNeg], axis: 1)

        let posIt = model([inpRefs], t: tArr, context: .raw([contextCond]), seqLen: seqLen, ropeCosSin: rope)[0]
        let posI = model([inpRefs], t: tArr, context: .raw([contextNull]), seqLen: seqLen, ropeCosSin: rope)[0]
        let neg = model([inpZero], t: tArr, context: .raw([contextNull]), seqLen: seqLen, ropeCosSin: rope)[0]

        let noisePred = neg + Float(guideImg) * (posI - neg) + Float(guideText) * (posIt - posI)
        let stepped = sched.step(
            modelOutput: noisePred.expandedDimensions(axis: 0), timestep: t,
            sample: latent.expandedDimensions(axis: 0))
        latent = stepped.squeezed(axis: 0)
        eval(latent)
        MLX.Memory.clearCache()
        WanDebug.stats("denoise step \(i + 1)/\(steps)", latent)  // WAN_DEBUG_STATS (latent divergence/zeroing)
        try onStep?(i, steps, latent)
    }
    return PhantomReference.stripRefs(latent, k: k)  // [16, F, h, w]
}
