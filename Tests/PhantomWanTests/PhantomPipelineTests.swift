import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import PhantomWan
import WanCore

/// End-to-end smoke for `PhantomPipeline.s2v`: umT5 encode → ref-encode → 3-forward S2V sample →
/// streaming decode. Each component is parity-gated/smoke-tested; this confirms the relay wires and
/// yields finite frames. CPU stream, tiny dims (umT5 fp32 load dominates the wall time).
final class PhantomPipelineTests: XCTestCase {
    static let dist = URL(fileURLWithPath:
        "/Volumes/DEV_ARCHIVE/phantom-wan-mlx/dist/Phantom-Wan-1.3B-MLX")

    func testS2VRelayRunsFinite() async throws {
        let needed = ["transformer-bf16.safetensors", "vae-encoder.safetensors",
                      "vae-decoder.safetensors", "t5_encoder.safetensors", "config.json"]
        for f in needed where !FileManager.default.fileExists(
            atPath: Self.dist.appendingPathComponent(f).path) {
            throw XCTSkip("Phantom checkpoint incomplete (missing \(f))")
        }
        try await Device.withDefaultDevice(Device(.cpu)) {
            let pipe = try await PhantomPipeline.fromPretrained(modelDir: Self.dist)

            // One synthetic reference subject @ 128² → 16² latent; 5 frames → 2 target latent frames.
            let ref = MLXRandom.uniform(low: -1.0, high: 1.0, [3, 128, 128])
            let out = try pipe.s2v(
                prompt: "a corgi running on a beach", referenceImages: [ref],
                width: 128, height: 128, numFrames: 5, steps: 2, seed: 0)
            eval(out)
            let mx = out.max().item(Float.self), mn = out.min().item(Float.self)
            print("[Phantom s2v smoke] frames \(out.shape) range [\(mn), \(mx)]")
            XCTAssertEqual(out.dim(0), 1, "batch")
            XCTAssertEqual(out.dim(1), 3, "channels")
            XCTAssertTrue(mx.isFinite && mn.isFinite, "non-finite frames")
            XCTAssertGreaterThanOrEqual(mn, -1.0001)
            XCTAssertLessThanOrEqual(mx, 1.0001)
        }
    }
}
