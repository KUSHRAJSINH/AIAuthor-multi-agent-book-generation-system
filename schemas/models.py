"""
Pydantic models for all structured outputs in the AIuthor system.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Tone
# ---------------------------------------------------------------------------

class ToneMetadata(BaseModel):
    """Tone configuration that drives every agent's style decisions."""
    name: str = Field(..., description="Tone name: conversational|academic|storyteller|motivational|witty")
    system_instruction: str = Field(..., description="System-level style directive injected into every prompt")
    vocabulary_level: str = Field(..., description="simple|intermediate|advanced|technical")
    sentence_variety: str = Field(..., description="short|mixed|long|complex")
    person: str = Field(..., description="first|second|third")
    formality: str = Field(..., description="casual|semi-formal|formal|academic")
    emotional_temperature: str = Field(..., description="cool|neutral|warm|passionate")
    example_opening: str = Field(default="", description="Sample opening sentence in this tone")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class GlossaryEntry(BaseModel):
    term: str
    definition: str
    chapter_introduced: int = 0
    tone_variant: Optional[str] = None


class CallbackReference(BaseModel):
    """A narrative callback — a concept/character/event introduced in one chapter
    that is referenced again in a later chapter."""
    callback_id: str
    introduced_in_chapter: int
    description: str
    referenced_in_chapters: List[int] = Field(default_factory=list)
    resolved: bool = False


class ChapterPlan(BaseModel):
    chapter_number: int
    title: str
    summary: str
    key_concepts: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    callbacks_to_introduce: List[str] = Field(default_factory=list)
    callbacks_to_resolve: List[str] = Field(default_factory=list)
    estimated_word_count: int = 800
    tone_notes: str = ""


class BookOutline(BaseModel):
    title: str
    subtitle: str = ""
    author_name: str = "AIuthor System"
    genre: str
    target_audience: str
    core_thesis: str
    narrative_arc: str
    chapters: List[ChapterPlan]
    glossary_seed: List[GlossaryEntry] = Field(default_factory=list)
    callback_index: List[CallbackReference] = Field(default_factory=list)
    tone: ToneMetadata
    foreword_notes: str = ""
    preface_notes: str = ""
    afterword_notes: str = ""
    about_author_notes: str = ""
    back_cover_copy: str = ""


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    source: str
    relevance_score: float = 0.0
    snippet: str = ""


class FactEntry(BaseModel):
    fact_id: str
    claim: str
    supported: bool
    citations: List[Citation] = Field(default_factory=list)
    softened_claim: Optional[str] = None
    chapter_number: int = 0


class ResearchPacket(BaseModel):
    chapter_number: int
    retrieved_chunks: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    fact_registry: List[FactEntry] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)
    bm25_results: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Writing & Editing
# ---------------------------------------------------------------------------

class ChapterContent(BaseModel):
    chapter_number: int
    title: str
    raw_content: str = ""
    humanized_content: str = ""
    edited_content: str = ""
    fact_checked_content: str = ""
    final_content: str = ""
    word_count: int = 0
    callbacks_used: List[str] = Field(default_factory=list)
    glossary_terms_used: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class CharacterEntry(BaseModel):
    name: str
    role: str
    first_appearance: int
    description: str
    traits: List[str] = Field(default_factory=list)


class ConceptEntry(BaseModel):
    concept: str
    explanation: str
    chapter_introduced: int
    complexity: str = "intermediate"


class ToneFingerprint(BaseModel):
    tone_name: str
    avg_sentence_length: float = 0.0
    contraction_ratio: float = 0.0
    question_frequency: float = 0.0
    exclamation_frequency: float = 0.0
    second_person_ratio: float = 0.0
    ai_tell_count: int = 0
    sample_sentences: List[str] = Field(default_factory=list)


class DecisionLogEntry(BaseModel):
    timestamp: str
    agent: str
    decision: str
    rationale: str
    chapter: int = 0


class CallbackIndex(BaseModel):
    callbacks: List[CallbackReference] = Field(default_factory=list)

    def get_open_callbacks(self) -> List[CallbackReference]:
        return [c for c in self.callbacks if not c.resolved]

    def get_callbacks_for_chapter(self, chapter_num: int) -> List[CallbackReference]:
        return [
            c for c in self.callbacks
            if chapter_num in c.referenced_in_chapters or c.introduced_in_chapter == chapter_num
        ]


class FactRegistry(BaseModel):
    entries: List[FactEntry] = Field(default_factory=list)

    def get_supported_facts(self) -> List[FactEntry]:
        return [f for f in self.entries if f.supported]

    def get_unsupported_facts(self) -> List[FactEntry]:
        return [f for f in self.entries if not f.supported]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class EvaluationMetric(BaseModel):
    metric_name: str
    score: float = Field(..., ge=0.0, le=10.0)
    rationale: str
    evidence: List[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    book_title: str
    tone: str
    chapter_count: int
    metrics: List[EvaluationMetric] = Field(default_factory=list)
    overall_score: float = 0.0
    llm_judge_verdict: str = ""
    recommendations: List[str] = Field(default_factory=list)
    generated_at: str = ""

    def compute_overall(self) -> float:
        if not self.metrics:
            return 0.0
        return round(sum(m.score for m in self.metrics) / len(self.metrics), 2)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class BookState(TypedDict, total=False):
    # Input
    user_brief: str
    tone_name: str
    chapter_count: int
    chapter_length: int          # target words per chapter
    author_name: str

    # Planning outputs
    outline: Optional[Dict[str, Any]]          # BookOutline.model_dump()

    # Research packets per chapter: {chapter_num: ResearchPacket.model_dump()}
    research_packets: Optional[Dict[int, Dict[str, Any]]]

    # Chapter content objects: {chapter_num: ChapterContent.model_dump()}
    chapters: Optional[Dict[int, Dict[str, Any]]]

    # Memory
    callback_index: Optional[Dict[str, Any]]   # CallbackIndex.model_dump()
    fact_registry: Optional[Dict[str, Any]]    # FactRegistry.model_dump()
    glossary: Optional[List[Dict[str, Any]]]   # List[GlossaryEntry.model_dump()]
    tone_fingerprint: Optional[Dict[str, Any]] # ToneFingerprint.model_dump()

    # Output paths
    docx_path: Optional[str]
    pdf_path: Optional[str]

    # Observability
    agent_logs: Optional[List[str]]
    errors: Optional[List[str]]
    current_chapter_index: int

    # Evaluation
    evaluation_report: Optional[Dict[str, Any]]
