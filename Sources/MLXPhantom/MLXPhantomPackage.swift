import Foundation
import MLX
import MLXToolKit
import PhantomWan
import WanCore

/// MLXEngine package: Phantom-Wan-1.3B (subject-to-video) over the canonical `textToVideo` surface.
/// Phantom is the consumer **subject-reference** image-grounding model — it composes up to 4 distinct
/// subjects from `T2VRequest.referenceImages` into one shot. The reference images are REQUIRED (it's
/// the conditioning signal); a `textToVideo` call without them is rejected.
///
/// Engine-owned lifecycle (C13, `@InferenceActor`): construct from `PhantomConfiguration`, page the
/// working set in with `load()`, drive `run(_:)`, reclaim with `unload()`. The non-`Sendable`
/// `PhantomPipeline` never crosses the actor boundary. Cancellation honored per denoising step.
///
/// The DiT is the stock wan-core `WanModel` (subject conditioning is input-assembly, not architecture);
/// the ref-encode + 3-forward sampler + streaming decode are parity-gated against the mlx_video oracle.
@InferenceActor
public final class MLXPhantomPackage: ModelPackage {
    public typealias Configuration = PhantomConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "Phantom-video/Phantom",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // DERIVED, pending a live in-app re-measure. Light consumer tier: the stock 1.3B DiT
                // (~9 GB fp32 with the 16-ch WanVAE), umT5 paged + evicted before denoise (§2.4). The
                // peak phase is the umT5 fp32 encode (~22 GB) — the 16 GB target needs fp8 umT5. Refs
                // are single-frame encodes (no encode wall), the decode streams. 3 DiT forwards/step
                // is a compute cost, not a memory one. Re-ground on the measured phys after the live run.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 28_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 24_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max  // pending the live re-measure (light tier — likely drops)
            ),
            specialties: [
                SpecialtyWeight(.general, strength: 0.5),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "phantom-1.3b-s2v",
                    summary: "Phantom-Wan-1.3B subject-to-video (MLX): supply 1–4 "
                        + "`T2VRequest.referenceImages` of distinct subjects + a prompt → a "
                        + "subject-consistent video. 832×480 native. References are REQUIRED.",
                    modes: [.quality, .fast]
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// The resident pipeline (stock WanModel + 16-ch WanVAE + tokenizer), paged in by `load()`.
    /// umT5 is NOT resident — paged in per request and evicted before denoise (§2.4).
    private var pipeline: PhantomPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        let directory: URL
        if let explicit = configuration.modelDirectory {
            directory = explicit
        } else {
            directory = try await WeightLoader.snapshotDownload(repoID: configuration.repo)
        }
        pipeline = try await PhantomPipeline.fromPretrained(modelDir: directory)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .textToVideo:
            guard let t2v = request as? T2VRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2VRequest", got: String(describing: type(of: request)))
            }
            return try await runS2V(t2v, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - Surface

    private func runS2V(_ request: T2VRequest, pipeline: PhantomPipeline) async throws -> T2VResponse {
        try Task.checkCancellation()
        guard let refs = request.referenceImages, !refs.isEmpty else {
            throw PackageError.configurationMismatch(
                expected: "T2VRequest.referenceImages (1–4 subject images — Phantom is subject-to-video)",
                got: "no reference images")
        }
        let numFrames = request.numFrames ?? 81
        let fps = request.fps ?? 16
        let width = request.width ?? 832
        let height = request.height ?? 480
        let steps = resolveSteps(mode: request.mode, steps: request.steps)
        let onStep: (Int, Int, MLXArray) throws -> Void = { _, _, _ in
            try Task.checkCancellation()  // C13: per-denoising-step cancellation
        }

        // Each reference Image → [3, H, W] in [-1, 1] (channels-first; decode then squeeze B,T).
        let refPixels = try refs.map {
            try decodeReferencePixels($0, width: width, height: height)
                .squeezed(axis: 0).squeezed(axis: 1)  // [1,3,1,H,W] → [3,H,W]
        }

        let frames = try pipeline.s2v(
            prompt: request.prompt, referenceImages: refPixels,
            negativePrompt: request.negativePrompt, width: width, height: height,
            numFrames: numFrames, steps: steps, guideImg: request.guidanceScale,
            seed: request.seed, onStep: onStep)
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return T2VResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
    }
}

extension MLXPhantomPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(MLXPhantomPackage.self)
    }
}
