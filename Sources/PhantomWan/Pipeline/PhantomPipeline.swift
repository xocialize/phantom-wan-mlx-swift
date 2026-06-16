// PhantomPipeline — the subject-to-video entry for Phantom-Wan-1.3B. Owns the component
// loads (stock wan-core `WanModel` + 16-ch WanVAE + umT5 tokenizer) and the (prompt + K
// reference images) → frames path. The DiT is the stock `WanModel` UNCHANGED; conditioning
// is the F+K input assembly (refs as trailing latent frames) + the 3-forward sampler. umT5 is
// paged in per request and evicted before denoise (§2.4). Refs are single-frame VAE encodes
// (cheap — no encode wall), and the decode streams (the wan-family lever).

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers
import WanCore

public final class PhantomPipeline: @unchecked Sendable {
    public let config: WanConfig
    /// Stock wan-core `WanModel` (subject conditioning is input-assembly, not architecture).
    public let model: WanModel
    /// 16-ch WanVAE — encode (reference images → trailing latent frames) + decode (output).
    public let vae: WanVAE
    /// Checkpoint dir — kept so umT5 can be (re)loaded per request and evicted before denoise.
    public let modelDir: URL
    public let tokenizer: any Tokenizer

    public init(
        config: WanConfig, model: WanModel, vae: WanVAE, modelDir: URL, tokenizer: any Tokenizer
    ) {
        self.config = config
        self.model = model
        self.vae = vae
        self.modelDir = modelDir
        self.tokenizer = tokenizer
    }

    /// Load from a flat checkpoint dir: `transformer-bf16.safetensors` + `vae-encoder.safetensors`
    /// + `vae-decoder.safetensors` + `t5_encoder.safetensors` + `config.json`. Tokenizer: umt5-xxl.
    public static func fromPretrained(
        modelDir: URL, ditDType: DType = .float32
    ) async throws -> PhantomPipeline {
        let config = try WanConfig.load(from: modelDir.appendingPathComponent("config.json"))

        // Stock WanModel — drop the stray `freqs` (rebuilt in-model).
        let model = WanModel(config)
        var dw = try WeightLoader.loadSafetensors(
            url: modelDir.appendingPathComponent("transformer-bf16.safetensors"))
        dw = dw.filter { $0.key != "freqs" }.mapValues { $0.asType(ditDType) }
        WeightLoader.materialize(dw)
        try model.update(parameters: ModuleParameters.unflattened(dw), verify: [.noUnusedKeys])
        eval(model.parameters())

        // 16-ch WanVAE — merge the separate encoder + decoder exports; wan-core computes
        // mean/std/inv_std from its (correct) Wan2.1 constants, so drop the exported stats.
        let vae = WanVAE(zDim: config.vaeZDim, encoder: true)
        let vaeWeights = try Device.withDefaultDevice(.cpu) { () -> [String: MLXArray] in
            let drop: Set<String> = ["mean", "std", "inv_std"]
            var all = try WeightLoader.loadSafetensors(
                url: modelDir.appendingPathComponent("vae-encoder.safetensors"))
            for (k, v) in try WeightLoader.loadSafetensors(
                url: modelDir.appendingPathComponent("vae-decoder.safetensors")) {
                all[k] = v
            }
            all = all.filter { !drop.contains($0.key) }.mapValues { $0.asType(.float32) }
            WeightLoader.materialize(all)
            return all
        }
        try vae.update(parameters: ModuleParameters.unflattened(vaeWeights), verify: [.noUnusedKeys])

        let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
        return PhantomPipeline(
            config: config, model: model, vae: vae, modelDir: modelDir, tokenizer: tokenizer)
    }

    // MARK: - umT5 (§2.4 post-encode eviction)

    private func loadTextEncoder() throws -> UMT5EncoderModel {
        let textEncoder = UMT5EncoderModel.fromConfig(config)
        let t5Weights = try WeightLoader.loadVerifiedSafetensors(
            url: modelDir.appendingPathComponent("t5_encoder.safetensors"),
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        ).mapValues { $0.asType(.float32) }
        WeightLoader.materialize(t5Weights)
        try textEncoder.update(
            parameters: ModuleParameters.unflattened(t5Weights), verify: [.noUnusedKeys])
        return textEncoder
    }

    func withTextEncoder<R>(_ body: (UMT5EncoderModel) throws -> R) throws -> R {
        var encoder: UMT5EncoderModel? = try loadTextEncoder()
        let result = try body(encoder!)
        encoder = nil
        MLX.GPU.clearCache()
        return result
    }

    // MARK: - Generation

    /// Subject-to-video: a prompt + K reference images (each `[3, H, W]` in [-1, 1], channels-first)
    /// → a subject-consistent video `[1, 3, T', H', W']` in [-1, 1]. Relay: umT5 encode→evict →
    /// ref-encode (single-frame, CPU stream) → 3-forward S2V sample → streaming decode.
    public func s2v(
        prompt: String,
        referenceImages: [MLXArray],
        negativePrompt: String? = nil,
        width: Int = 832,
        height: Int = 480,
        numFrames: Int = 81,
        steps: Int? = nil,
        guideImg: Double? = nil,
        guideText: Double? = nil,
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        precondition(!referenceImages.isEmpty, "Phantom s2v needs ≥1 reference image")
        let negative = negativePrompt ?? config.sampleNegPrompt

        // §2.4: page umT5 in, encode cond/uncond, evict before denoise.
        let (contextCond, contextNull) = try withTextEncoder { enc -> (MLXArray, MLXArray) in
            let c = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
            let n = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
            eval(c, n)
            return (c, n)
        }

        // Reference latents — single-frame encodes on the CPU stream (cheap; no encode wall).
        let refLatents = Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let r = PhantomReference.encodeReferences(vae: vae, refs: referenceImages)
            eval(r)
            return r
        }

        // Target latent geometry from the vae strides.
        let fLat = (numFrames - 1) / config.vaeStride[0] + 1
        let hLat = height / config.vaeStride[1]
        let wLat = width / config.vaeStride[2]

        let latent = try sampleS2V(
            model: model, refLatents: refLatents, contextCond: contextCond, contextNull: contextNull,
            config: config, fLatent: fLat, hLatent: hLat, wLatent: wLat,
            steps: steps ?? config.sampleSteps, shift: config.sampleShift,
            guideImg: guideImg ?? 5.0, guideText: guideText ?? 7.5, seed: seed, onStep: onStep)
        eval(latent)
        MLX.GPU.clearCache()

        return decodeLatent(latent)
    }

    /// Decode a channels-first latent `[C, T, h, w]` → frames `[1, 3, T', H', W']` in [-1, 1].
    /// Streaming decode (one latent chunk live) + the `Memory.cacheLimit` cap, on the CPU stream.
    public func decodeLatent(_ latent: MLXArray) -> MLXArray {
        let prevCacheLimit = Memory.cacheLimit
        let capMB = ProcessInfo.processInfo.environment["DECODE_CACHE_MB"].flatMap { Int($0) } ?? 2048
        Memory.cacheLimit = capMB * 1_000_000
        defer { Memory.cacheLimit = prevCacheLimit }
        MLX.GPU.clearCache()
        return Device.withDefaultDevice(.cpu) {
            let video = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0), chunkLat: 1)
            eval(video)
            return video
        }
    }
}
