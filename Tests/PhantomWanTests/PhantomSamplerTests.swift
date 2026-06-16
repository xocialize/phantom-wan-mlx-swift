import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import PhantomWan
import WanCore

/// Smoke gate for the 3-forward S2V sampler: the forward is parity-gated (`PhantomForwardParity`)
/// and the scheduler is wan-core's, so the loop is correct-by-construction — this confirms it wires
/// (re-clamp, CFG combine, strip) and yields a finite, correctly-shaped target latent. Small dims +
/// synthetic refs/context on the CPU stream.
final class PhantomSamplerTests: XCTestCase {
    static let dist = URL(fileURLWithPath:
        "/Volumes/DEV_ARCHIVE/phantom-wan-mlx/dist/Phantom-Wan-1.3B-MLX")

    func testSampleS2VRunsFinite() throws {
        let weights = Self.dist.appendingPathComponent("transformer-bf16.safetensors")
        if !FileManager.default.fileExists(atPath: weights.path) {
            throw XCTSkip("Phantom weights not present")
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            let config = try WanConfig.load(from: Self.dist.appendingPathComponent("config.json"))
            let model = WanModel(config)
            let w = try WeightLoader.loadSafetensors(url: weights)
                .filter { $0.key != "freqs" }.mapValues { $0.asType(.float32) }
            try model.update(parameters: ModuleParameters.unflattened(w), verify: [.noUnusedKeys])
            eval(model.parameters())

            let (fLat, k, h, wd) = (2, 1, 16, 16)              // F=2 target + K=1 ref → 3 frames
            let refLatents = MLXRandom.normal([16, k, h, wd])  // synthetic clean ref latent
            let ctx = MLXRandom.normal([8, config.textDim])
            eval(refLatents, ctx)

            let out = try sampleS2V(
                model: model, refLatents: refLatents, contextCond: ctx, contextNull: ctx * 0,
                config: config, fLatent: fLat, hLatent: h, wLatent: wd,
                steps: 2, shift: 5.0, guideImg: 5.0, guideText: 7.5, seed: 0)
            eval(out)
            let mx = out.max().item(Float.self), mn = out.min().item(Float.self)
            print("[Phantom sampler smoke] out \(out.shape) range [\(mn), \(mx)]")
            XCTAssertEqual(out.shape, [16, fLat, h, wd], "sampler strips the K ref tail → [16, F, h, w]")
            XCTAssertTrue(mx.isFinite && mn.isFinite, "non-finite sampler output")
        }
    }
}
