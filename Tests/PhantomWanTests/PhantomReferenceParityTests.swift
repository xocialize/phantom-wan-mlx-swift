import Foundation
import MLX
import MLXNN
import XCTest

@testable import PhantomWan
import WanCore

/// Gate the ref conditioning: `PhantomReference.encodeReferences` (single-frame 16-ch WanVAE
/// encode → trailing latent) vs the oracle `encode_references`. The WanVAE encode is parity-locked,
/// so this should be tight. Golden: `phantom_parity` `in_ref` [1,3,1,128,128] → `g_ref_latent`
/// [1,16,1,16,16].
final class PhantomReferenceParityTests: XCTestCase {
    static let root = URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/phantom-wan-mlx")
    var dist: URL { Self.root.appendingPathComponent("dist/Phantom-Wan-1.3B-MLX") }
    var parityDir: URL { Self.root.appendingPathComponent("phantom_parity") }
    private func golden(_ n: String) throws -> MLXArray {
        try loadNumpy(url: parityDir.appendingPathComponent("\(n).npy"))
    }

    func testEncodeReferencesMatchesOracle() throws {
        let vaeW = dist.appendingPathComponent("vae-encoder.safetensors")
        if !FileManager.default.fileExists(atPath: vaeW.path) {
            throw XCTSkip("Phantom VAE encoder not present")
        }
        try Device.withDefaultDevice(Device(.cpu)) {
            let vae = WanVAE(zDim: 16, encoder: true)
            // wan-core's WanVAE computes mean/std/inv_std from constants (not loadable params).
            let drop: Set<String> = ["mean", "std", "inv_std"]
            let w = try WeightLoader.loadSafetensors(url: vaeW)
                .filter { !drop.contains($0.key) }
                .mapValues { $0.asType(.float32) }
            try vae.update(parameters: ModuleParameters.unflattened(w), verify: [.noUnusedKeys])
            eval(vae.parameters())

            let inRef = try golden("in_ref")          // [1, 3, 1, 128, 128]
            let gRef = try golden("g_ref_latent")     // [1, 16, 1, 16, 16]
            let ref = inRef.squeezed()                // [3, 128, 128] (encodeReferences re-adds B,T)

            let z = PhantomReference.encodeReferences(vae: vae, refs: [ref])  // [16, 1, 16, 16]
            eval(z)
            XCTAssertEqual(z.shape, [16, 1, 16, 16], "ref latent shape")
            let maxd = abs(z - gRef[0]).max().item(Float.self)
            print("[Phantom ref-encode] vs oracle: max-abs=\(maxd) shape=\(z.shape)")
            // ~0.008: the Phantom export ships bf16 mean/std/inv_std, while wan-core normalizes with
            // its fp64 Wan2.1 constants (the CORRECT stats, proven bit-exact by Bernini/VACE). So this
            // is the oracle's bf16-stat rounding, not a wan-core error — structurally correct, accepted
            // for the functional pass (conditioning latent). Bound gates shape + regressions.
            XCTAssertLessThan(maxd, 0.01, "ref-encode max-abs \(maxd)")
        }
    }
}
