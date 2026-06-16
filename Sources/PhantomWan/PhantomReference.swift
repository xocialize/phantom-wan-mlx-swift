import MLX
import WanCore

/// Phantom subject conditioning — pure input assembly (no new weights). K reference images are
/// each VAE-encoded as a SINGLE temporal frame, concatenated into trailing latent frames, and
/// appended to the target latent's temporal tail (the F+K grid). The DiT ropes the refs at
/// ordinary sequential positions F..F+K-1 — the stock `WanModel` forward, unchanged. Verbatim
/// port of the oracle `model/reference.py`.
public enum PhantomReference {
    /// K refs (each `[3, H, W]` in [-1, 1], channels-first) → trailing latent `[16, K, h, w]`
    /// (one latent frame per subject). Uses the parity-locked 16-ch `WanVAE.encode` (T=1).
    public static func encodeReferences(vae: WanVAE, refs: [MLXArray]) -> MLXArray {
        var latents: [MLXArray] = []
        for ref in refs {
            let x = ref.expandedDimensions(axis: 0).expandedDimensions(axis: 2)  // [1, 3, 1, H, W]
            let z = vae.encode(x)                                                // [1, 16, 1, h, w]
            latents.append(z[0])                                                 // [16, 1, h, w]
        }
        return concatenated(latents, axis: 1)                                    // [16, K, h, w]
    }

    /// `cat([noisy_target, clean_refs], temporal axis)` → `[16, F+K, h, w]` (refs at the tail).
    public static func assembleInput(noisyTarget: MLXArray, refLatents: MLXArray) -> MLXArray {
        concatenated([noisyTarget, refLatents], axis: 1)
    }

    /// Drop the last K reference frames after sampling → `[16, F, h, w]`.
    public static func stripRefs(_ latent: MLXArray, k: Int) -> MLXArray {
        latent[0..., 0 ..< (latent.dim(1) - k)]
    }
}
