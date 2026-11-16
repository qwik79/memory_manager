"""Pluggable background distillation worker for incremental context compression.

Architecture mirrors speculative decoding: a cheap *draft* distiller produces a
result immediately while a heavier *refine* distiller optionally upgrades it
asynchronously.  The ``BackgroundWorker`` daemon thread pulls message deltas
from a queue and calls whichever ``Distiller`` implementation is configured.

Available distillers
--------------------
``HeuristicDistiller``
    Zero-dependency TF-IDF extractive summariser.  Runs in ~0 ms.  Use as the
    sole distiller for AI-free operation, or as the draft in
    ``SpeculativeDistiller``.

``LLMDistiller``
    Thin adapter over ``OllamaSummarizer`` for incremental delta processing.
    Forces CPU inference (``num_gpu=0``) so VRAM stays with the Core model.

``CascadeDistiller``
    Try each distiller in order; return the first that succeeds.
    Example: ``CascadeDistiller([LLMDistiller(), HeuristicDistiller()])``

``SpeculativeDistiller``
    Heuristic draft applied immediately; LLM refinement queued on an inner
    thread.  The background worker (and callers of ``flush()``) get the draft
    instantly.  When the LLM finishes it calls the refinement callback so the
    session state is silently upgraded.
"""

from __future__ import annotations

import logging
import math
import queue
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical empty result — every distiller should return this schema
# ---------------------------------------------------------------------------


def _empty_result() -> dict[str, Any]:
    return {
        "summary": "",
        "decisions": [],
        "next_steps": [],
        "facts": {},
        "tier": "extend",
        "notes_for_core": "",
        "confidence": 0.0,
    }


# ---------------------------------------------------------------------------
# Distiller protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Distiller(Protocol):
    """Incremental distiller interface.

    Unlike :class:`~memory_manager.summarizer.Summarizer`, which operates on
    a full batch of ``MemoryEntry`` dicts, a ``Distiller`` processes a *delta*
    of raw ``{role, content}`` message dicts plus the current pinned summary
    as context.
    """

    def distill(
        self, messages: list[dict[str, Any]], prev_summary: str = ""
    ) -> dict[str, Any]:
        """Return a structured distillation dict with the canonical schema."""


# ---------------------------------------------------------------------------
# HeuristicDistiller — zero dependencies, always available
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    "the a an and or but in on at to for of with by from is are was were "
    "be been being have has had do does did will would could should may might "
    "shall that this these those it its we you he she they their our my your "
    "his her i me him us them what which who when where how if then so just "
    "also get got can not no as up out about into like than more very".split()
)

_DECISION_RE = re.compile(
    r"(?:decision|decided|going with|will use|chosen|choosing|selected|"
    r"\[pin\]|pinned)[\s:]+([^\n.!?]{10,120})",
    re.IGNORECASE,
)
_NEXT_STEP_RE = re.compile(
    r"(?:TODO|next up|next step|next:|will need|need to|going to|should|"
    r"plan to|✓|→|\[ \]|\[\])\s*:?\s*([^\n.!?]{5,100})",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?:[./][\w./\\-]+\.(?:py|js|ts|go|rs|yaml|json|toml|md|sh|txt))")
_CONFIG_RE = re.compile(r'(?:port|url|host|addr|key|model|endpoint)\s*[:=]\s*[\w./:-]+', re.IGNORECASE)


class HeuristicDistiller:
    """Pure-algorithmic extractive distiller using TF-IDF sentence scoring.

    Produces results in ~0 ms with no external dependencies.  Quality is
    lower than an LLM but it is always available, never blocks, and is
    sufficiently accurate for the draft phase of speculative distillation.
    """

    is_instant: bool = True  # runs in microseconds — safe to call inline

    def __init__(self, max_summary_sentences: int = 5, max_items: int = 5) -> None:
        self.max_summary_sentences = max_summary_sentences
        self.max_items = max_items

    def distill(
        self, messages: list[dict[str, Any]], prev_summary: str = ""
    ) -> dict[str, Any]:
        if not messages:
            return _empty_result()

        content = "\n".join(
            f"{m.get('role', 'unknown').upper()}: {m.get('content', '')}"
            for m in messages
        )
        if not content.strip():
            return _empty_result()

        sentences = self._split_sentences(content)
        scores = self._tfidf_scores(sentences)
        ranked = sorted(zip(sentences, scores), key=lambda x: -x[1])
        top = [s.strip() for s, _ in ranked[: self.max_summary_sentences] if s.strip()]

        decisions = list(dict.fromkeys(
            m.group(1).strip() for m in _DECISION_RE.finditer(content)
        ))[: self.max_items]
        next_steps = list(dict.fromkeys(
            m.group(1).strip() for m in _NEXT_STEP_RE.finditer(content)
        ))[: self.max_items]
        facts = self._extract_facts(content)
        tier = self._infer_tier(content, decisions)
        notes = self._check_risks(content)

        return {
            "summary": " ".join(top),
            "decisions": decisions,
            "next_steps": next_steps,
            "facts": facts,
            "tier": tier,
            "notes_for_core": notes,
            "confidence": 0.6,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        return [
            t.lower()
            for t in re.split(r"\W+", text)
            if len(t) > 2 and t.lower() not in _STOP_WORDS
        ]

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        return [s.strip() for s in raw if len(s.strip()) > 20]

    def _tfidf_scores(self, sentences: list[str]) -> list[float]:
        if not sentences:
            return []
        sent_tokens = [self._tokenize(s) for s in sentences]
        doc_freq: Counter[str] = Counter()
        for tokens in sent_tokens:
            doc_freq.update(set(tokens))
        n = len(sentences)
        scores: list[float] = []
        for tokens in sent_tokens:
            if not tokens:
                scores.append(0.0)
                continue
            tf = Counter(tokens)
            score = sum(
                freq * math.log((n + 1) / (doc_freq[term] + 1))
                for term, freq in tf.items()
            )
            scores.append(score / len(tokens))
        return scores

    def _extract_facts(self, text: str) -> dict[str, Any]:
        paths = sorted(set(_PATH_RE.findall(text)))[: self.max_items]
        configs = sorted(set(_CONFIG_RE.findall(text)))[: self.max_items]
        facts: dict[str, Any] = {}
        if paths:
            facts["paths"] = paths
        if configs:
            facts["config"] = configs
        return facts

    def _infer_tier(self, text: str, decisions: list[str]) -> str:
        if decisions or re.search(r"\b(critical|important|must|always|never)\b", text, re.I):
            return "pin"
        if re.search(r"\b(resolved|done|fixed|complete|merged|closed)\b", text, re.I):
            return "archive"
        return "extend"

    def _check_risks(self, text: str) -> str:
        warnings: list[str] = []
        if re.search(r"\b(contradict|conflict|inconsist|was wrong|actually)\b", text, re.I):
            warnings.append("Possible contradiction detected in conversation.")
        if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", text):
            warnings.append("Unresolved TODOs/FIXMEs present.")
        return " ".join(warnings)


# ---------------------------------------------------------------------------
# LLMDistiller — wraps OllamaSummarizer for incremental delta processing
# ---------------------------------------------------------------------------


class LLMDistiller:
    """Ollama-backed LLM distiller for incremental delta processing.

    Defaults to CPU-only (``num_gpu=0``) to leave VRAM for the core model.
    Set ``num_gpu=-1`` (all layers on GPU) or a positive integer if you have
    headroom — e.g. a 24 GB card running a ≤14 B core model can spare 1–2 GB
    for a 1.5 B worker.
    """

    is_instant: bool = False  # network call — always async

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        timeout: float = 90.0,
        num_gpu: int = 0,
    ) -> None:
        from .summarizer import HeuristicSummarizer, OllamaSummarizer

        self._summarizer = OllamaSummarizer(
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            fallback=HeuristicSummarizer(),
            num_gpu=num_gpu,
        )

    def distill(
        self, messages: list[dict[str, Any]], prev_summary: str = ""
    ) -> dict[str, Any]:
        entries = [
            {"entry_type": m.get("role", "unknown"), "content": m.get("content", "")}
            for m in messages
        ]
        context = {"pinned_summary": prev_summary}
        return self._summarizer.summarize_entries(entries, context)


# ---------------------------------------------------------------------------
# OpenAIDistiller — wraps OpenAISummarizer for any /v1/chat/completions server
# ---------------------------------------------------------------------------


class OpenAIDistiller:
    """Distiller for any OpenAI-compatible local server.

    Wraps :class:`~openai_summarizer.OpenAISummarizer` to cover LM Studio,
    LocalAI, vLLM, llama.cpp server, text-generation-webui, and any other
    server exposing ``/v1/chat/completions``.

    Same interface as :class:`LLMDistiller` — drop-in replacement.
    """

    is_instant: bool = False  # network call — always async

    def __init__(
        self,
        model: str = "local-model",
        endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
        timeout: float = 90.0,
    ) -> None:
        from .openai_summarizer import OpenAISummarizer

        self._summarizer = OpenAISummarizer(
            model=model,
            endpoint=endpoint,
            timeout=timeout,
        )

    def distill(
        self, messages: list[dict[str, Any]], prev_summary: str = ""
    ) -> dict[str, Any]:
        entries = [
            {"entry_type": m.get("role", "unknown"), "content": m.get("content", "")}
            for m in messages
        ]
        context = {"pinned_summary": prev_summary}
        return self._summarizer.summarize_entries(entries, context)


# ---------------------------------------------------------------------------
# CascadeDistiller — try each distiller in order
# ---------------------------------------------------------------------------


class CascadeDistiller:
    """Try each distiller in turn; return the first result with a non-empty summary.

    Example::

        distiller = CascadeDistiller([
            LLMDistiller(model="qwen2.5:1.5b"),
            HeuristicDistiller(),
        ])
    """

    def __init__(self, distillers: list[Distiller]) -> None:
        if not distillers:
            raise ValueError("CascadeDistiller requires at least one distiller.")
        self._distillers = distillers

    @property
    def is_instant(self) -> bool:
        """True only when every distiller in the chain is instant."""
        return all(getattr(d, "is_instant", False) for d in self._distillers)

    def distill(
        self, messages: list[dict[str, Any]], prev_summary: str = ""
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for distiller in self._distillers:
            try:
                result = distiller.distill(messages, prev_summary)
                if result.get("summary"):
                    return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.debug(
                    "Distiller %s failed, trying next: %s",
                    type(distiller).__name__,
                    exc,
                )
        logger.warning(
            "All distillers in cascade failed (last: %s). Returning empty result.",
            last_exc,
        )
        return _empty_result()


# ---------------------------------------------------------------------------
# SpeculativeDistiller — heuristic draft immediately, LLM refinement async
# ---------------------------------------------------------------------------


class SpeculativeDistiller:
    """Speculative-decoding-style distiller.

    :meth:`distill` returns the heuristic draft synchronously (near-instant).
    An inner thread simultaneously runs the *refine* distiller.  When it
    finishes it calls the registered refinement callback so the session state
    is silently upgraded — exactly like a verifier accepting a speculative
    draft.

    Usage::

        distiller = SpeculativeDistiller(
            draft=HeuristicDistiller(),
            refine=LLMDistiller(model="qwen2.5:1.5b"),
        )
        # BackgroundWorker registers the callback automatically.
    """

    is_instant: bool = False  # refine leg is always async

    def __init__(self, draft: Distiller, refine: Distiller) -> None:
        self._draft = draft
        self._refine = refine
        self._callback: Callable[[dict[str, Any]], None] | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mem-speculative"
        )

    def set_refinement_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register the function called when LLM refinement completes."""
        self._callback = callback

    def distill(
        self, messages: list[dict[str, Any]], prev_summary: str = ""
    ) -> dict[str, Any]:
        draft_result = self._draft.distill(messages, prev_summary)
        if self._callback is not None:
            self._executor.submit(self._refine_and_notify, messages, prev_summary)
        return draft_result

    def stop(self) -> None:
        """Shutdown the refinement thread pool."""
        self._executor.shutdown(wait=False)

    def _refine_and_notify(
        self, messages: list[dict[str, Any]], prev_summary: str
    ) -> None:
        try:
            refined = self._refine.distill(messages, prev_summary)
            if self._callback is not None and refined.get("summary"):
                self._callback(refined)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Speculative refinement failed (draft was kept): %s", exc)


# ---------------------------------------------------------------------------
# BackgroundWorker — daemon thread that drives continuous distillation
# ---------------------------------------------------------------------------


class BackgroundWorker:
    """Continuous background distillation daemon.

    Push message deltas after each exchange with :meth:`push`.  The worker
    thread drains the queue, batches pending deltas together, and calls the
    configured ``Distiller``.  Call :meth:`flush` before reading session state
    at trigger time to ensure the last delta has been processed.

    Parameters
    ----------
    distiller:
        Any :class:`Distiller` implementation.  If it is a
        :class:`SpeculativeDistiller`, the refinement callback is registered
        automatically.
    on_result:
        Called with the distillation dict when the worker produces a result.
        Typically ``manager.apply_distillation``.
    get_summary:
        Zero-argument callable returning the current pinned summary string.
        Typically ``manager.get_pinned_summary``.
    """

    def __init__(
        self,
        distiller: Distiller,
        on_result: Callable[[dict[str, Any]], None],
        get_summary: Callable[[], str],
    ) -> None:
        self._distiller = distiller
        self._on_result = on_result
        self._get_summary = get_summary
        self._queue: queue.Queue[list[dict[str, Any]]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mem-background-worker"
        )
        # Wire speculative refinement callback if supported.
        if isinstance(distiller, SpeculativeDistiller):
            distiller.set_refinement_callback(on_result)
        self._thread.start()

    def push(self, messages: list[dict[str, Any]]) -> None:
        """Enqueue a delta of new messages for background distillation."""
        if messages:
            self._queue.put(messages)

    def flush(self, timeout: float = 30.0) -> None:
        """Block until the queue is empty (all deltas processed).

        Call this before compacting hot→warm at trigger time to ensure the
        background distillation of the most recent exchange is applied to
        session state first.
        """
        try:
            self._queue.join()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BackgroundWorker.flush timed out or failed: %s", exc)

    def stop(self) -> None:
        """Signal the worker thread to stop and wait for it to exit."""
        self._stop.set()
        self._thread.join(timeout=5.0)
        if hasattr(self._distiller, "stop"):
            self._distiller.stop()  # type: ignore[union-attr]

    def _run(self) -> None:
        while not self._stop.is_set():
            # Block up to 2 s so the thread wakes cleanly on stop.
            try:
                first = self._queue.get(timeout=2.0)
            except queue.Empty:
                continue

            # Drain any additional pending deltas and process them together.
            batch: list[list[dict[str, Any]]] = [first]
            while not self._queue.empty():
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            messages = [m for delta in batch for m in delta]
            prev = self._get_summary()

            try:
                result = self._distiller.distill(messages, prev)
                if result.get("summary"):
                    self._on_result(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("BackgroundWorker distillation error: %s", exc)
            finally:
                for _ in batch:
                    self._queue.task_done()
