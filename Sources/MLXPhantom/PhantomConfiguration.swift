import Foundation
import WanCore
import MLXToolKit

/// Init-time configuration for `MLXPhantomPackage` (C9): which variant + where the flat checkpoint
/// lives. Per-request prompt/refs/size ride the canonical `T2VRequest`, not here.
public struct PhantomConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    public var repo: String
    public var revision: String?
    public var quant: Quant
    /// Resolved local checkpoint folder. Environment-specific → excluded from `Codable`.
    public var modelDirectory: URL?
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "Phantom-video/Phantom",
        revision: String? = nil,
        quant: Quant = .bf16,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the resolved flat checkpoint into the OS file
/// cache before `load()`'s GPU evals, so a cold load-time `eval` never faults weights off
/// slow/external storage inside a live Metal command buffer (the cold-load GPU watchdog,
/// `kIOGPUCommandBufferCallbackErrorTimeout`). Phantom-Wan-1.3B is a light backbone (unlikely to
/// bite), but the conformance is family-uniform. The whole resolved `modelDirectory` is paged; only
/// the config knows the path, execution is the engine's (`WeightPrewarmer`, best-effort). Nil on the
/// HF-download path → no-op.
extension PhantomConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] { [modelDirectory].compactMap { $0 } }
}
