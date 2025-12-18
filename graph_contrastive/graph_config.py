"""
Edge Configuration Registry for Graph-Contrastive Learning.

This module defines the edge taxonomy and configuration for different data sources.
To add a new data source (e.g., ProofWiki, Mathlib), simply add new entries to
EDGE_REGISTRY without modifying any other code.

Edge Types:
    0: SE_Answer      - StackExchange Question -> Answer
    1: SE_Tag         - StackExchange Post -> Tag (reserved for future)
    2: SE_Duplicate   - StackExchange Question <-> Duplicate Question
    3: SE_Linked      - StackExchange Post -> Linked Post
    4: ArXiv_Proof    - ArXiv Theorem -> Proof
    5: ArXiv_Dep      - ArXiv Statement -> Dependency
    6: ArXiv_Ordinal  - ArXiv Sequential statements in paper
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import IntEnum


class EdgeType(IntEnum):
    """Enumeration of all edge types across data sources."""
    SE_ANSWER = 0
    SE_TAG = 1
    SE_DUPLICATE = 2
    SE_LINKED = 3
    ARXIV_PROOF = 4
    ARXIV_DEP = 5
    ARXIV_ORDINAL = 6


class SourceType(IntEnum):
    """Enumeration of data sources."""
    STACKEXCHANGE = 0
    ARXIV = 1
    PROOFWIKI = 2  # Reserved for future
    MATHLIB = 3    # Reserved for future


@dataclass
class EdgeConfig:
    """Configuration for a single edge type."""
    edge_type_id: int
    name: str
    source_type: int
    src_prompt: str           # Prompt prefix for source node
    tgt_prompt: str           # Prompt prefix for target node
    weight: float             # Default weight for loss weighting
    bidirectional: bool       # Whether to create reverse edge
    description: str = ""     # Human-readable description


# =============================================================================
# EDGE REGISTRY
# =============================================================================
# This is the single source of truth for edge configuration.
# To add a new data source, simply add new EdgeConfig entries here.

EDGE_REGISTRY: Dict[int, EdgeConfig] = {
    # -------------------------------------------------------------------------
    # StackExchange Edges
    # -------------------------------------------------------------------------
    EdgeType.SE_ANSWER: EdgeConfig(
        edge_type_id=EdgeType.SE_ANSWER,
        name="SE_Answer",
        source_type=SourceType.STACKEXCHANGE,
        src_prompt="Question:",
        tgt_prompt="Answer:",
        weight=1.5,  # Higher weight for accepted answers (handled in preprocessing)
        bidirectional=False,
        description="Question -> Answer (accepted=2.0, voted=1.0)"
    ),

    EdgeType.SE_TAG: EdgeConfig(
        edge_type_id=EdgeType.SE_TAG,
        name="SE_Tag",
        source_type=SourceType.STACKEXCHANGE,
        src_prompt="Classify:",
        tgt_prompt="Topic:",
        weight=1.0,
        bidirectional=False,
        description="Post -> Tag classification"
    ),

    EdgeType.SE_DUPLICATE: EdgeConfig(
        edge_type_id=EdgeType.SE_DUPLICATE,
        name="SE_Duplicate",
        source_type=SourceType.STACKEXCHANGE,
        src_prompt="Question:",
        tgt_prompt="Duplicate:",
        weight=1.0,
        bidirectional=True,  # Duplicates are symmetric
        description="Question <-> Duplicate Question"
    ),

    EdgeType.SE_LINKED: EdgeConfig(
        edge_type_id=EdgeType.SE_LINKED,
        name="SE_Linked",
        source_type=SourceType.STACKEXCHANGE,
        src_prompt="Post:",
        tgt_prompt="Related:",
        weight=0.8,
        bidirectional=False,
        description="Post -> Linked Post (hyperlink reference)"
    ),

    # -------------------------------------------------------------------------
    # ArXiv Edges
    # -------------------------------------------------------------------------
    EdgeType.ARXIV_PROOF: EdgeConfig(
        edge_type_id=EdgeType.ARXIV_PROOF,
        name="ArXiv_Proof",
        source_type=SourceType.ARXIV,
        src_prompt="Theorem:",
        tgt_prompt="Proof:",
        weight=2.0,  # High weight - direct semantic relationship
        bidirectional=False,
        description="Theorem/Lemma -> Its Proof"
    ),

    EdgeType.ARXIV_DEP: EdgeConfig(
        edge_type_id=EdgeType.ARXIV_DEP,
        name="ArXiv_Dep",
        source_type=SourceType.ARXIV,
        src_prompt="Statement:",
        tgt_prompt="Uses:",
        weight=1.0,
        bidirectional=False,
        description="Statement -> Referenced/Used Statement"
    ),

    EdgeType.ARXIV_ORDINAL: EdgeConfig(
        edge_type_id=EdgeType.ARXIV_ORDINAL,
        name="ArXiv_Ordinal",
        source_type=SourceType.ARXIV,
        src_prompt="Context:",
        tgt_prompt="Next:",
        weight=0.5,  # Lower weight - weaker semantic signal
        bidirectional=False,
        description="Sequential statements in paper (u_i -> u_{i+1})"
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_edge_config(edge_type_id: int) -> EdgeConfig:
    """Get configuration for an edge type."""
    if edge_type_id not in EDGE_REGISTRY:
        raise ValueError(f"Unknown edge type: {edge_type_id}")
    return EDGE_REGISTRY[edge_type_id]


def get_prompts(edge_type_id: int) -> tuple:
    """Get (src_prompt, tgt_prompt) for an edge type."""
    config = get_edge_config(edge_type_id)
    return config.src_prompt, config.tgt_prompt


def get_weight(edge_type_id: int) -> float:
    """Get default weight for an edge type."""
    return get_edge_config(edge_type_id).weight


def format_text_with_prompt(text: str, prompt: str) -> str:
    """Format text with its prompt prefix."""
    return f"{prompt} {text}"


# =============================================================================
# Node Type Mappings (for ArXiv)
# =============================================================================

ARXIV_ENV_MAPPING = {
    # Theorem-like environments (can have proofs)
    'theorem': 'theorem',
    'thm': 'theorem',
    'lemma': 'lemma',
    'lem': 'lemma',
    'proposition': 'proposition',
    'prop': 'proposition',
    'corollary': 'corollary',
    'cor': 'corollary',
    'claim': 'claim',

    # Definition-like environments
    'definition': 'definition',
    'def': 'definition',
    'defn': 'definition',

    # Other environments
    'remark': 'remark',
    'rem': 'remark',
    'example': 'example',
    'exa': 'example',
    'assumption': 'assumption',
    'proof': 'proof',
}


def normalize_arxiv_env(env_raw: str) -> str:
    """Normalize ArXiv environment name to canonical form."""
    env_lower = env_raw.lower()
    return ARXIV_ENV_MAPPING.get(env_lower, env_lower)


def is_provable_env(env: str) -> bool:
    """Check if an environment type can have an associated proof."""
    provable = {'theorem', 'lemma', 'proposition', 'corollary', 'claim'}
    return normalize_arxiv_env(env) in provable
