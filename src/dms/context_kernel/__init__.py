"""Creative context kernel: source-aware memory and external knowledge substrate."""

from dms.context_kernel.assembly import (
    ContextAssembler,
    CreativeContextPacketConfig,
    format_creative_context_packet_markdown,
)
from dms.context_kernel.audit import (
    audit_external_vs_artifact_conflicts,
    export_external_entity_links,
    export_external_timeline_claims,
)
from dms.context_kernel.json_schema import (
    creative_context_json_schemas,
    write_creative_context_json_schemas,
)
from dms.context_kernel.kernel import (
    CreativeMemoryKernel,
    ExternalKnowledgeKernel,
    artifact_record_to_context_item,
    import_artifact_store_items,
    import_reference_library_items,
    reference_record_to_context_item,
)
from dms.context_kernel.providers import (
    ChromaEmbeddingProvider,
    ContextLLMProvider,
    EmbeddingProvider,
    FastAPIReranker,
    FastAPIRerankerConfig,
    LLMClientProvider,
    OpenAICompatibleEmbeddingConfig,
    OpenAICompatibleLLMConfig,
    RerankerProvider,
    ScoreReranker,
    openai_compatible_llm_provider,
)
from dms.context_kernel.reconciliation import (
    ReconciliationDecision,
    apply_reconciliation_decision,
    decide_reconciliation_deterministic,
    reconcile_items_deterministic,
)
from dms.context_kernel.schema import (
    CreativeContextItem,
    CreativeScope,
    EntityPatch,
    EvidenceRef,
    SourceRecord,
    SourceUnit,
    stable_id,
)
from dms.context_kernel.store import CreativeContextStore
from dms.context_kernel.vector_index import (
    ContextChromaIndexConfig,
    build_context_chroma_index,
    search_context_chroma_index,
)

__all__ = [
    "ContextAssembler",
    "ContextChromaIndexConfig",
    "CreativeContextItem",
    "CreativeContextPacketConfig",
    "CreativeContextStore",
    "CreativeMemoryKernel",
    "CreativeScope",
    "ChromaEmbeddingProvider",
    "ContextLLMProvider",
    "EmbeddingProvider",
    "EntityPatch",
    "EvidenceRef",
    "ExternalKnowledgeKernel",
    "FastAPIReranker",
    "FastAPIRerankerConfig",
    "LLMClientProvider",
    "OpenAICompatibleEmbeddingConfig",
    "OpenAICompatibleLLMConfig",
    "RerankerProvider",
    "ReconciliationDecision",
    "ScoreReranker",
    "SourceRecord",
    "SourceUnit",
    "artifact_record_to_context_item",
    "audit_external_vs_artifact_conflicts",
    "apply_reconciliation_decision",
    "build_context_chroma_index",
    "creative_context_json_schemas",
    "decide_reconciliation_deterministic",
    "export_external_entity_links",
    "export_external_timeline_claims",
    "format_creative_context_packet_markdown",
    "import_artifact_store_items",
    "import_reference_library_items",
    "openai_compatible_llm_provider",
    "reference_record_to_context_item",
    "reconcile_items_deterministic",
    "search_context_chroma_index",
    "stable_id",
    "write_creative_context_json_schemas",
]
