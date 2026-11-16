"""Tiered memory manager for extending LLM context with local memory tiers."""

from .active_context import ActiveContext, ContextItem
from .background_worker import (
    BackgroundWorker,
    CascadeDistiller,
    Distiller,
    HeuristicDistiller,
    LLMDistiller,
    OpenAIDistiller,
    SpeculativeDistiller,
)
from .cold_store import ColdStoreManager
from .embeddings import make_embed_fn, make_ollama_embed_fn
from .hot_buffer import HotBufferManager
from .model_utils import ensure_embedding_model, ensure_model
from .models import MemoryEntry, MemorySession, MemoryTierConfig, RetrievalResult, WarmSection
from .ollama_integration import LLMIntegration, OllamaIntegration
from .openai_summarizer import OpenAISummarizer
from .orchestrator import TieredMemoryManager
from .scoring import EntryIndex, auto_classify_importance, compute_relevance_score
from .security import SecurityManager, TamperDetectedError, get_security_manager
from .summarizer import HeuristicSummarizer, OllamaSummarizer
from .trigger import OffloadTrigger, TriggerConfig
from .warm_store import WarmStoreManager

__all__ = [
    "ActiveContext",
    "BackgroundWorker",
    "CascadeDistiller",
    "ColdStoreManager",
    "ContextItem",
    "Distiller",
    "EntryIndex",
    "HeuristicDistiller",
    "HeuristicSummarizer",
    "HotBufferManager",
    "LLMDistiller",
    "LLMIntegration",
    "MemoryEntry",
    "MemorySession",
    "MemoryTierConfig",
    "OffloadTrigger",
    "OllamaIntegration",
    "OllamaSummarizer",
    "OpenAIDistiller",
    "OpenAISummarizer",
    "RetrievalResult",
    "SecurityManager",
    "SpeculativeDistiller",
    "TamperDetectedError",
    "TieredMemoryManager",
    "TriggerConfig",
    "WarmSection",
    "WarmStoreManager",
    "auto_classify_importance",
    "compute_relevance_score",
    "ensure_embedding_model",
    "ensure_model",
    "get_security_manager",
    "make_embed_fn",
    "make_ollama_embed_fn",
]
