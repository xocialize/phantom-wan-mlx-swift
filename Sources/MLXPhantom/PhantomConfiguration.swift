import Foundation
import WanCore
import MLXToolKit

/// Init-time configuration for `MLXPhantomPackage` (C9): which variant + where the flat checkpoint
/// lives. Per-request prompt/refs/size ride the canonical `T2VRequest`, not here.
public struct PhantomConfiguration: PackageConfiguration, ModelStorable {
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
