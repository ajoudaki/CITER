#!/usr/bin/env python3
r"""
Build theorem/lemma/etc. dependency DAGs from a JSONL of LaTeX sources (multiprocessing).

Edge semantics:
- edge u -> v means: v depends on u (u is a prerequisite / dependency of v)

Normalization:
- node["env"] is a CANONICAL KIND (theorem/lemma/definition/...)
- node["env_raw"] is the raw LaTeX environment name from \begin{...}

Label containment:
- Any \label{...} inside a node environment is treated as an alias for that node.

Proof attachment:
- If proof header resolves to a node: attach there.
- Otherwise: attach only to the most recent *proof-bearing* node (theorem/lemma/prop/cor/claim [+ optionally definition]),
  not to remarks/examples.

Preprocessing:
- Removes blocks swallowed by "\forget ... \forgotten".

CHANGED (primary label selection):
- The node's primary label/id is now chosen as the first \label{...} that is NOT inside a math-like sub-environment
  (equation/align/subequations/...). This avoids picking equation labels inside a proposition as the node label.
  If no such label exists, we fall back to the first label (to preserve stable IDs when the only label is inside math).

NEW (progress):
- Shows progress while queueing input records and while writing output records (tqdm if available).

NEW (proof nodes):
- Each \begin{proof}...\end{proof} becomes a node with env="proof".
- We add an edge: proof_node -> attached_statement_node.
"""

import argparse
import json
import os
import re
import sys
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from collections import deque

# ---------------------------
# Constants
# ---------------------------

CANON_KINDS = [
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "claim",
    "definition",
    "example",
    "remark",
    "assumption",
    "proof",  # NEW
]
CANON_KIND_SET = set(CANON_KINDS)

# Kinds eligible for fallback proof attachment
PROOF_TARGET_KINDS: Set[str] = {
    "theorem", "lemma", "proposition", "corollary", "claim",
    # "definition",  # remove if you do not want proofs to attach to definitions by fallback
}

DEFAULT_ENV_NAMES = [
    "theorem", "theorems",
    "lemma", "lemmas",
    "proposition", "propositions",
    "corollary", "corollaries",
    "claim", "claims",
    "definition", "definitions",
    "example", "examples",
    "remark", "remarks",
    "assumption", "assumptions",
]

DEFAULT_REFCMDS = ["ref", "cref", "Cref", "autoref", "nameref", "namecref", "eqref"]

DISPLAY_TOKEN_TO_KIND = {
    "theorem": "theorem",
    "theorems": "theorem",
    "lemma": "lemma",
    "lemmas": "lemma",
    "proposition": "proposition",
    "propositions": "proposition",
    "corollary": "corollary",
    "corollaries": "corollary",
    "claim": "claim",
    "claims": "claim",
    "definition": "definition",
    "definitions": "definition",
    "example": "example",
    "examples": "example",
    "remark": "remark",
    "remarks": "remark",
    "assumption": "assumption",
    "assumptions": "assumption",
}

ABBREV_ENV_TO_KIND = {
    "thm": "theorem",
    "thmx": "theorem",
    "thrm": "theorem",
    "theo": "theorem",
    "lem": "lemma",
    "lmm": "lemma",
    "lma": "lemma",
    "lm": "lemma",
    "prop": "proposition",
    "propo": "proposition",
    "propos": "proposition",
    "prp": "proposition",
    "cor": "corollary",
    "coro": "corollary",
    "corol": "corollary",
    "crl": "corollary",
    "rmk": "remark",
    "rem": "remark",
    "defn": "definition",
    "defi": "definition",
    "dfn": "definition",
    "def": "definition",
    "eg": "example",
    "ex": "example",
    "exa": "example",
    "exmp": "example",
    "ass": "assumption",
    "asp": "assumption",
    "assum": "assumption",
}

# environments treated as "math containers" for primary-label selection
MATH_ENV_MASK: Set[str] = {
    "equation", "align", "eqnarray", "gather", "multline", "flalign", "alignat",
    "subequations", "split",
}

# ---------------------------
# Data model
# ---------------------------

@dataclass
class Node:
    ordinal: int
    id: str
    env: str          # canonical kind
    env_raw: str      # raw LaTeX environment name
    file: str
    file_index: int
    start_offset: int
    label: Optional[str]   # primary label
    text: str
    labels: List[str]      # all labels inside env block

@dataclass
class WorkerConfig:
    envs: List[str]
    include_forward: bool
    emit_forward: bool
    lowercase_labels: bool
    label_map: Optional[Dict[str, str]]
    ref_commands: List[str]
    fuzzy: bool
    fuzzy_threshold: float
    include_statement_refs: bool
    ordered_output: bool


# ---------------------------
# Preprocessing utilities
# ---------------------------

def remove_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)

def _clean_letters(s: str) -> str:
    return re.sub(r"[^a-z]+", "", s.lower())

def remove_forget_blocks(text: str) -> str:
    pat = re.compile(r"\\forget\b.*?\\forgotten\b", re.DOTALL)
    while True:
        m = pat.search(text)
        if not m:
            break
        text = pat.sub("", text, count=1)
    return text


# ---------------------------
# Env inference / normalization
# ---------------------------

def infer_env_kind_map_from_newtheorem(all_text: str) -> Dict[str, str]:
    env_to_kind: Dict[str, str] = {}
    pat = re.compile(
        r"\\newtheorem\*?\s*"
        r"\{(?P<env>[A-Za-z@]+)\}\s*"
        r"(?:\[[^\]]*\]\s*)?"
        r"\{(?P<display>[^}]+)\}\s*"
        r"(?:\[[^\]]*\]\s*)?",
        re.IGNORECASE
    )

    for m in pat.finditer(all_text):
        env_name = m.group("env").strip()
        display = m.group("display").strip()
        disp_norm = _clean_letters(display)

        best_kind = None
        best_len = -1
        for token, kind in DISPLAY_TOKEN_TO_KIND.items():
            token_norm = _clean_letters(token)
            if token_norm and token_norm in disp_norm:
                if len(token_norm) > best_len:
                    best_len = len(token_norm)
                    best_kind = kind

        if best_kind is not None:
            env_to_kind[env_name] = best_kind

    return env_to_kind

def normalize_env_kind(env_raw: str, env_to_kind: Dict[str, str]) -> str:
    if env_raw in env_to_kind:
        return env_to_kind[env_raw]

    low = env_raw.lower()
    if low in DISPLAY_TOKEN_TO_KIND:
        return DISPLAY_TOKEN_TO_KIND[low]

    low_clean = _clean_letters(low)
    if low_clean in ABBREV_ENV_TO_KIND:
        return ABBREV_ENV_TO_KIND[low_clean]

    for kind in CANON_KINDS:
        if kind in low_clean:
            return kind

    for abbr, kind in ABBREV_ENV_TO_KIND.items():
        if abbr in low_clean:
            return kind

    return low


# ---------------------------
# Ref parsing
# ---------------------------

def build_ref_pattern(ref_cmds: List[str]) -> re.Pattern:
    cmds = DEFAULT_REFCMDS[:]
    for c in ref_cmds:
        if c and isinstance(c, str) and c not in cmds:
            cmds.append(c)
    part = "|".join(re.escape(c) for c in cmds)
    return re.compile(rf"\\(?:{part})\s*\{{([^}}]*)\}}")

def extract_refs_from_text(text: Optional[str], refpat: re.Pattern) -> List[str]:
    if not text:
        return []
    refs: List[str] = []
    for m in refpat.finditer(text):
        content = m.group(1)
        for part in content.split(","):
            p = part.strip()
            if p:
                refs.append(p)
    seen: Set[str] = set()
    out: List[str] = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ---------------------------
# Ref-like macro learning
# ---------------------------

def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i].isspace():
        i += 1
    return i

def _parse_command_name_at(s: str, i: int) -> Tuple[Optional[str], int]:
    if i >= len(s) or s[i] != "\\":
        return None, i
    j = i + 1
    while j < len(s) and (s[j].isalpha() or s[j] == "@"):
        j += 1
    if j == i + 1:
        return None, i
    return s[i + 1:j], j

def _parse_brace_group(s: str, i: int) -> Tuple[Optional[str], int]:
    if i >= len(s) or s[i] != "{":
        return None, i
    depth = 1
    j = i + 1
    start = j
    while j < len(s):
        ch = s[j]
        if ch == "\\" and j + 1 < len(s):
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:j], j + 1
        j += 1
    return None, i

def _parse_macro_name_spec(s: str, i: int) -> Tuple[Optional[str], int]:
    i = _skip_ws(s, i)
    if i >= len(s):
        return None, i

    if s[i] == "{":
        inner, j = _parse_brace_group(s, i)
        if inner is None:
            return None, i
        inner_str = inner.strip()
        if inner_str.startswith("\\"):
            name, _ = _parse_command_name_at(inner_str, 0)
            return name, j
        return None, j

    if s[i] == "\\":
        name, j = _parse_command_name_at(s, i)
        return name, j

    return None, i

def _skip_balanced_optional_brackets(s: str, i: int) -> int:
    if i >= len(s) or s[i] != "[":
        return i
    depth = 0
    j = i
    while j < len(s):
        ch = s[j]
        if ch == "\\" and j + 1 < len(s):
            j += 2
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return i

def _ref_use_pattern_from_cmds(ref_cmds: List[str]) -> Optional[re.Pattern]:
    cmds: List[str] = []
    for c in ref_cmds:
        if c and isinstance(c, str) and c not in cmds:
            cmds.append(c)
    if not cmds:
        return None
    part = "|".join(re.escape(c) for c in cmds)
    return re.compile(rf"\\(?:{part})\*?\s*\{{[^}}]*#1[^}}]*\}}")

def _infer_ref_like_macros_newcommand(all_text: str, ref_use_pat: re.Pattern) -> List[str]:
    out: List[str] = []
    tok = re.compile(r"\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand)\*?\b")
    for m in tok.finditer(all_text):
        i = _skip_ws(all_text, m.end())
        name, i = _parse_macro_name_spec(all_text, i)
        if not name:
            continue

        i = _skip_ws(all_text, i)
        if i < len(all_text) and all_text[i] == "[":
            i = _skip_balanced_optional_brackets(all_text, i)

        i = _skip_ws(all_text, i)
        if i < len(all_text) and all_text[i] == "[":
            i = _skip_balanced_optional_brackets(all_text, i)

        i = _skip_ws(all_text, i)
        body, _ = _parse_brace_group(all_text, i)
        if body is None:
            continue

        if ref_use_pat.search(body):
            out.append(name)
    return out

def _infer_ref_like_macros_xparse(all_text: str, ref_use_pat: re.Pattern) -> List[str]:
    out: List[str] = []
    tok = re.compile(r"\\(?:NewDocumentCommand|RenewDocumentCommand|ProvideDocumentCommand|DeclareDocumentCommand)\b")
    for m in tok.finditer(all_text):
        i = _skip_ws(all_text, m.end())
        name, i = _parse_macro_name_spec(all_text, i)
        if not name:
            continue

        i = _skip_ws(all_text, i)
        sig, i2 = _parse_brace_group(all_text, i)
        if sig is None:
            continue

        i2 = _skip_ws(all_text, i2)
        body, _ = _parse_brace_group(all_text, i2)
        if body is None:
            continue

        if ref_use_pat.search(body):
            out.append(name)
    return out

def _infer_ref_like_macros_def(all_text: str, ref_use_pat: re.Pattern) -> List[str]:
    out: List[str] = []
    tok = re.compile(r"\\(?:def|edef|gdef|xdef)\b")
    for m in tok.finditer(all_text):
        i = _skip_ws(all_text, m.end())
        name, i2 = _parse_command_name_at(all_text, i)
        if not name:
            continue

        j = i2
        while j < len(all_text):
            ch = all_text[j]
            if ch == "\\" and j + 1 < len(all_text):
                j += 2
                continue
            if ch == "{":
                break
            j += 1
        if j >= len(all_text) or all_text[j] != "{":
            continue

        params = all_text[i2:j]
        if "#1" not in params:
            continue

        body, _ = _parse_brace_group(all_text, j)
        if body is None:
            continue

        if ref_use_pat.search(body):
            out.append(name)
    return out

def infer_ref_like_macros(all_text: str, known_ref_cmds: List[str]) -> List[str]:
    r"""
    Learn ref-like wrapper macros defined in the LaTeX source itself.
    """
    ref_use_pat = _ref_use_pattern_from_cmds(known_ref_cmds)
    if ref_use_pat is None or not all_text:
        return []

    found: List[str] = []
    found.extend(_infer_ref_like_macros_newcommand(all_text, ref_use_pat))
    found.extend(_infer_ref_like_macros_xparse(all_text, ref_use_pat))
    found.extend(_infer_ref_like_macros_def(all_text, ref_use_pat))

    seen: Set[str] = set()
    dedup: List[str] = []
    for name in found:
        if name and name not in seen:
            seen.add(name)
            dedup.append(name)
    return dedup


# ---------------------------
# Environment / proof block scanning
# ---------------------------

def find_theorem_blocks(text: str, envs: List[str]) -> List[Tuple[int, int, str, str]]:
    blocks: List[Tuple[int, int, str, str]] = []
    for env in envs:
        pat = re.compile(
            rf"\\begin\{{{re.escape(env)}\*?\}}\s*.*?\s*\\end\{{{re.escape(env)}\*?\}}",
            re.DOTALL
        )
        for m in pat.finditer(text):
            blocks.append((m.start(), m.end(), env, m.group(0)))
    blocks.sort(key=lambda t: t[0])
    return blocks

def find_proof_blocks(text: str) -> List[Tuple[int, int, Optional[str], str, str]]:
    pat = re.compile(
        r"\\begin\{proof\*?\}(?:\[(?P<header>[^\]]*)\])?(?P<body>.*?)\\end\{proof\*?\}",
        re.DOTALL
    )
    out: List[Tuple[int, int, Optional[str], str, str]] = []
    for m in pat.finditer(text):
        out.append((m.start(), m.end(), m.group("header"), m.group("body"), m.group(0)))
    return out

def extract_all_labels_with_pos(block: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for m in re.finditer(r"\\label\{([^}]+)\}", block):
        lab = m.group(1).strip()
        if lab:
            out.append((lab, m.start()))
    seen: Set[str] = set()
    dedup: List[Tuple[str, int]] = []
    for lab, pos in out:
        if lab not in seen:
            seen.add(lab)
            dedup.append((lab, pos))
    return dedup

def extract_all_labels(block: str) -> List[str]:
    return [lab for (lab, _) in extract_all_labels_with_pos(block)]

def _normalize_env_name_for_mask(env_name: str) -> str:
    e = env_name.strip()
    if e.endswith("*"):
        e = e[:-1]
    return e.lower()

def compute_mask_spans(block: str, mask_envs: Set[str]) -> List[Tuple[int, int]]:
    r"""
    Compute spans [start,end) for masked environments inside the block using a simple \begin/\end stack.
    """
    spans: List[Tuple[int, int]] = []
    tok = re.compile(r"\\(?P<kind>begin|end)\{(?P<env>[^}]+)\}")
    stack: List[Tuple[str, int]] = []

    for m in tok.finditer(block):
        kind = m.group("kind")
        env = _normalize_env_name_for_mask(m.group("env"))
        if kind == "begin":
            stack.append((env, m.start()))
        else:
            if not stack:
                continue
            top_env, top_start = stack[-1]
            if top_env != env:
                found_idx = None
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == env:
                        found_idx = i
                        break
                if found_idx is None:
                    continue
                top_env, top_start = stack[found_idx]
                stack = stack[:found_idx + 1]

            stack.pop()
            end_pos = m.end()
            if env in mask_envs:
                spans.append((top_start, end_pos))

    spans.sort()
    return spans

def pos_in_spans(pos: int, spans: List[Tuple[int, int]]) -> bool:
    for a, b in spans:
        if a <= pos < b:
            return True
    return False

def extract_primary_label(block: str) -> Optional[str]:
    r"""
    Choose primary label as the first \label{...} that is NOT inside a math-like environment span.
    If none, fall back to first label.
    """
    labs = extract_all_labels_with_pos(block)
    if not labs:
        return None

    spans = compute_mask_spans(block, MATH_ENV_MASK)

    for lab, pos in labs:
        if not pos_in_spans(pos, spans):
            return lab

    return labs[0][0]

def extract_env_inner_text(block: str, env: str) -> str:
    s = block.strip()
    m_begin = re.match(rf"\\begin\{{{re.escape(env)}\*?\}}", s)
    if not m_begin:
        body = s
        body = re.sub(rf"^\s*\\begin\{{{re.escape(env)}\*?\}}\s*", "", body, flags=re.DOTALL)
        body = re.sub(rf"\\end\{{{re.escape(env)}\*?\}}\s*$", "", body, flags=re.DOTALL)
    else:
        i = m_begin.end()
        while i < len(s) and s[i].isspace():
            i += 1
        if i < len(s) and s[i] == "[":
            j = _skip_balanced_optional_brackets(s, i)
            if j != i:
                i = j
            while i < len(s) and s[i].isspace():
                i += 1
        m_end = re.search(rf"\\end\{{{re.escape(env)}\*?\}}\s*$", s, flags=re.DOTALL)
        body = s[i:m_end.start()] if m_end else s[i:]

    body = re.sub(r"\\label\{[^}]+\}", "", body)
    return body.strip()

def scan_events(cleaned_texts: List[Tuple[str, str]], envs: List[str]) -> List[Dict]:
    events: List[Dict] = []
    for fi, (name, text) in enumerate(cleaned_texts):
        for start, end, env, block in find_theorem_blocks(text, envs):
            events.append({"type": "thm", "file_index": fi, "start": start, "env": env, "block": block, "file": name})
        for start, end, hdr, body, block in find_proof_blocks(text):
            events.append({"type": "prf", "file_index": fi, "start": start, "header": hdr, "body": body, "block": block, "file": name})
    events.sort(key=lambda e: (e["file_index"], e["start"]))
    return events


# ---------------------------
# Graph helpers
# ---------------------------

def is_earlier(a_file_index: int, a_start: int, b_file_index: int, b_start: int) -> bool:
    return (a_file_index, a_start) < (b_file_index, b_start)

def dedupe_edges(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

def topological_sort_ids(ids: List[str], edges: List[Tuple[str, str]]) -> Tuple[List[str], bool, Dict[str, List[str]]]:
    indeg: Dict[str, int] = {i: 0 for i in ids}
    adj: Dict[str, List[str]] = {i: [] for i in ids}

    for u, v in edges:
        adj.setdefault(u, [])
        adj.setdefault(v, [])
        indeg.setdefault(u, 0)
        indeg.setdefault(v, 0)
        adj[u].append(v)
        indeg[v] += 1

    q = deque([i for i in ids if indeg.get(i, 0) == 0])
    topo: List[str] = []
    count = 0
    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
        count += 1

    return topo, (count == len(ids)), adj


# ---------------------------
# Label indices & ref resolution
# ---------------------------

def normalize_label_case(label: Optional[str], lowercase: bool) -> Optional[str]:
    if label is None:
        return None
    return label.casefold() if lowercase else label

def canonicalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())

def label_tokens(label: str) -> Set[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label)
    parts = re.split(r"[:_/.\- ]+", s)
    return {p.strip().lower() for p in parts if p.strip()}

def build_label_indices(
    nodes: List[Node],
    lowercase_labels: bool,
    label_map: Optional[Dict[str, str]]
) -> Tuple[Dict[str, int], Dict[str, Set[int]], Dict[int, Set[str]]]:
    direct: Dict[str, int] = {}
    canonical_map: Dict[str, Set[int]] = {}
    tokens_map: Dict[int, Set[str]] = {}

    for idx, n in enumerate(nodes):
        for lab in (n.labels or []):
            norm = normalize_label_case(lab, lowercase_labels)
            if norm is None:
                continue
            if norm not in direct:
                direct[norm] = idx
            canonical_map.setdefault(canonicalize_label(norm), set()).add(idx)
            tokens_map.setdefault(idx, set()).update(label_tokens(norm))

    if label_map:
        for alias, target in label_map.items():
            alias_norm = normalize_label_case(alias, lowercase_labels)
            target_norm = normalize_label_case(target, lowercase_labels)
            if alias_norm is None or target_norm is None:
                continue
            if target_norm in direct and alias_norm not in direct:
                direct[alias_norm] = direct[target_norm]
                canonical_map.setdefault(canonicalize_label(alias_norm), set()).add(direct[target_norm])
                tokens_map.setdefault(direct[target_norm], set()).update(label_tokens(alias_norm))

    return direct, canonical_map, tokens_map

def resolve_ref(
    raw_ref: str,
    direct_map: Dict[str, int],
    canonical_map: Dict[str, Set[int]],
    tokens_map: Dict[int, Set[str]],
    lowercase_labels: bool,
    label_map: Optional[Dict[str, str]],
    fuzzy: bool,
    fuzzy_threshold: float
) -> Tuple[Optional[int], str]:
    mapped = label_map.get(raw_ref) if label_map else None
    ref1 = mapped if mapped else raw_ref
    norm = normalize_label_case(ref1, lowercase_labels)
    if norm in direct_map:
        return direct_map[norm], "direct"

    if not fuzzy:
        return None, "unknown"

    can = canonicalize_label(norm)
    cand_set = canonical_map.get(can)
    if cand_set and len(cand_set) == 1:
        return list(cand_set)[0], "canonical"

    ref_toks = label_tokens(norm)
    best_idx = None
    best_score = 0.0
    for idx, toks in tokens_map.items():
        if not toks:
            continue
        inter = len(ref_toks & toks)
        union = len(ref_toks | toks)
        if union == 0:
            continue
        score = inter / union
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None and best_score >= fuzzy_threshold:
        return best_idx, "fuzzy"
    return None, "unknown"


# ---------------------------
# Pass 1: statement nodes
# ---------------------------

def pass1_collect_nodes(events: List[Dict], env_to_kind: Dict[str, str]) -> Tuple[List[Node], Dict[Tuple[int, int], int]]:
    nodes: List[Node] = []
    key_to_idx: Dict[Tuple[int, int], int] = {}
    ordinal = 0

    for ev in events:
        if ev["type"] != "thm":
            continue

        env_raw = ev["env"]
        env_kind = normalize_env_kind(env_raw, env_to_kind)

        block = ev["block"]
        all_labels = extract_all_labels(block)
        primary_label = extract_primary_label(block)
        text = extract_env_inner_text(block, env_raw)

        ordinal += 1
        basename = os.path.basename(ev["file"])
        auto_id = f"{env_raw}#{ordinal}@{basename}"
        node_id = primary_label if primary_label else auto_id

        node = Node(
            ordinal=ordinal,
            id=node_id,
            env=env_kind,
            env_raw=env_raw,
            file=ev["file"],
            file_index=ev["file_index"],
            start_offset=ev["start"],
            label=primary_label,
            text=text,
            labels=all_labels,
        )
        nodes.append(node)
        key_to_idx[(ev["file_index"], ev["start"])] = len(nodes) - 1

    return nodes, key_to_idx


# ---------------------------
# Pass 2: proofs -> edges (+ proof nodes)
# ---------------------------

def _make_proof_node_id(proof_ordinal: int, proof_file: str, proof_start: int) -> str:
    # NEW: Stable-ish per record, avoids colliding with typical LaTeX labels.
    basename = os.path.basename(proof_file)
    return f"proof#{proof_ordinal}@{basename}:{proof_start}"

def select_proof_target(
    hdr_refs_raw: List[str],
    pending_proof_stack: List[int],
    proof_assigned: Dict[int, bool],
    direct_map: Dict[str, int],
    canonical_map: Dict[str, Set[int]],
    tokens_map: Dict[int, Set[str]],
    config: WorkerConfig
) -> Tuple[Optional[int], Optional[str]]:
    for raw in hdr_refs_raw:
        idx_cand, method = resolve_ref(
            raw_ref=raw,
            direct_map=direct_map,
            canonical_map=canonical_map,
            tokens_map=tokens_map,
            lowercase_labels=config.lowercase_labels,
            label_map=config.label_map,
            fuzzy=config.fuzzy,
            fuzzy_threshold=config.fuzzy_threshold
        )
        if idx_cand is not None:
            return idx_cand, method

    while pending_proof_stack:
        cand = pending_proof_stack.pop()
        if not proof_assigned.get(cand, False):
            return cand, "fallback-proofable"

    return None, None

def maybe_add_edge(
    target_node: Node,
    dep_node: Node,
    proof_pos: Tuple[int, int],
    edges: List[Tuple[str, str]],
    forward_edges: List[Tuple[str, str]],
    config: WorkerConfig,
    skipped_forward_ref_counter: List[int]
) -> None:
    proof_fi, proof_start = proof_pos
    earlier = is_earlier(dep_node.file_index, dep_node.start_offset, proof_fi, proof_start)

    edge = (dep_node.id, target_node.id)  # dependency -> dependent

    if earlier or config.include_forward:
        edges.append(edge)
        if (not earlier) and config.emit_forward:
            forward_edges.append(edge)
        return

    skipped_forward_ref_counter[0] += 1
    if config.emit_forward:
        forward_edges.append(edge)

def build_edges(
    events: List[Dict],
    nodes: List[Node],
    key_to_idx: Dict[Tuple[int, int], int],
    config: WorkerConfig
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, int], int, List[Node]]:  # CHANGED: returns proof_nodes
    refpat = build_ref_pattern(config.ref_commands)

    edges: List[Tuple[str, str]] = []
    forward_edges: List[Tuple[str, str]] = []
    unknown_refs: Dict[str, int] = {}
    skipped_forward = [0]

    proof_assigned = {i: False for i in range(len(nodes))}
    pending_proof_stack: List[int] = []

    direct_map, canonical_map, tokens_map = build_label_indices(nodes, config.lowercase_labels, config.label_map)

    # NEW: collect proof nodes here (so we can also emit proof->target edges)
    proof_nodes: List[Node] = []
    proof_count = 0
    proof_ordinal_base = len(nodes)

    for ev in events:
        if ev["type"] == "thm":
            idx = key_to_idx[(ev["file_index"], ev["start"])]
            if nodes[idx].env in PROOF_TARGET_KINDS:
                pending_proof_stack.append(idx)
            continue

        # ev["type"] == "prf"
        hdr = ev.get("header") or ""
        body = ev.get("body") or ""
        hdr_refs_raw = extract_refs_from_text(hdr, refpat)

        target_idx, _ = select_proof_target(
            hdr_refs_raw=hdr_refs_raw,
            pending_proof_stack=pending_proof_stack,
            proof_assigned=proof_assigned,
            direct_map=direct_map,
            canonical_map=canonical_map,
            tokens_map=tokens_map,
            config=config
        )

        # NEW: create a proof node for every proof block
        proof_count += 1
        proof_ordinal = proof_ordinal_base + proof_count
        proof_id = _make_proof_node_id(proof_ordinal, ev["file"], ev["start"])
        proof_node = Node(
            ordinal=proof_ordinal,
            id=proof_id,
            env="proof",
            env_raw="proof",
            file=ev["file"],
            file_index=ev["file_index"],
            start_offset=ev["start"],
            label=None,
            text=body.strip(),
            labels=[],
        )
        proof_nodes.append(proof_node)

        if target_idx is None:
            # Preserve the prior behavior: if we can't attach, we skip dependency edges (but keep proof node).
            for raw in hdr_refs_raw:
                idx_dep, _ = resolve_ref(
                    raw_ref=raw,
                    direct_map=direct_map,
                    canonical_map=canonical_map,
                    tokens_map=tokens_map,
                    lowercase_labels=config.lowercase_labels,
                    label_map=config.label_map,
                    fuzzy=config.fuzzy,
                    fuzzy_threshold=config.fuzzy_threshold
                )
                if idx_dep is None:
                    unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
            continue

        proof_assigned[target_idx] = True
        target_node = nodes[target_idx]
        proof_pos = (ev["file_index"], ev["start"])

        # NEW: proof node -> attached statement node
        edges.append((proof_node.id, target_node.id))

        for raw in hdr_refs_raw:
            idx_dep, _ = resolve_ref(
                raw_ref=raw,
                direct_map=direct_map,
                canonical_map=canonical_map,
                tokens_map=tokens_map,
                lowercase_labels=config.lowercase_labels,
                label_map=config.label_map,
                fuzzy=config.fuzzy,
                fuzzy_threshold=config.fuzzy_threshold
            )
            if idx_dep is None:
                unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
                continue
            if idx_dep == target_idx:
                continue
            dep_node = nodes[idx_dep]
            maybe_add_edge(target_node, dep_node, proof_pos, edges, forward_edges, config, skipped_forward)

        seen_ids: Set[str] = set()
        body_refs_raw = extract_refs_from_text(body, refpat)
        for raw in body_refs_raw:
            idx_dep, _ = resolve_ref(
                raw_ref=raw,
                direct_map=direct_map,
                canonical_map=canonical_map,
                tokens_map=tokens_map,
                lowercase_labels=config.lowercase_labels,
                label_map=config.label_map,
                fuzzy=config.fuzzy,
                fuzzy_threshold=config.fuzzy_threshold
            )
            if idx_dep is None:
                unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
                continue
            if idx_dep == target_idx:
                continue

            dep_node = nodes[idx_dep]
            if dep_node.id in seen_ids:
                continue
            seen_ids.add(dep_node.id)

            maybe_add_edge(target_node, dep_node, proof_pos, edges, forward_edges, config, skipped_forward)

    return dedupe_edges(edges), dedupe_edges(forward_edges), unknown_refs, skipped_forward[0], proof_nodes  # CHANGED

def add_statement_edges(
    nodes: List[Node],
    edges: List[Tuple[str, str]],
    forward_edges: List[Tuple[str, str]],
    unknown_refs: Dict[str, int],
    config: WorkerConfig,
    skipped_forward: List[int]
) -> None:
    if not config.include_statement_refs or not nodes:
        return

    refpat = build_ref_pattern(config.ref_commands)
    direct_map, canonical_map, tokens_map = build_label_indices(nodes, config.lowercase_labels, config.label_map)

    for idx_src, n in enumerate(nodes):
        refs_raw = extract_refs_from_text(n.text, refpat)
        for raw in refs_raw:
            idx_dep, _ = resolve_ref(
                raw_ref=raw,
                direct_map=direct_map,
                canonical_map=canonical_map,
                tokens_map=tokens_map,
                lowercase_labels=config.lowercase_labels,
                label_map=config.label_map,
                fuzzy=config.fuzzy,
                fuzzy_threshold=config.fuzzy_threshold
            )
            if idx_dep is None:
                unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
                continue
            if idx_dep == idx_src:
                continue

            dep_node = nodes[idx_dep]
            earlier = is_earlier(dep_node.file_index, dep_node.start_offset, n.file_index, n.start_offset)
            edge = (dep_node.id, n.id)  # dependency -> dependent

            if earlier or config.include_forward:
                edges.append(edge)
            else:
                skipped_forward[0] += 1
                if config.emit_forward:
                    forward_edges.append(edge)


# ---------------------------
# Record-level processing
# ---------------------------

def merge_envs(envs: List[str], inferred: List[str]) -> List[str]:
    out = envs[:]
    seen = set(out)
    for e in inferred:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

def process_record(filenames: List[str], latex_list: List[str], config: WorkerConfig) -> Dict:
    if len(filenames) != len(latex_list):
        raise ValueError("filenames and latex arrays must have equal length")

    cleaned_texts: List[Tuple[str, str]] = []
    for i in range(len(filenames)):
        name = filenames[i]
        text = latex_list[i] if latex_list[i] is not None else ""
        t1 = remove_comments(text)
        t2 = remove_forget_blocks(t1)
        cleaned_texts.append((name, t2))

    all_text = "\n".join(t for _, t in cleaned_texts)

    base_ref_cmds_for_detection = DEFAULT_REFCMDS[:] + list(config.ref_commands or [])
    inferred_ref_cmds = infer_ref_like_macros(all_text, base_ref_cmds_for_detection)
    ref_cmds_local = list(config.ref_commands or [])
    if inferred_ref_cmds:
        for c in inferred_ref_cmds:
            if c and c not in ref_cmds_local and c not in DEFAULT_REFCMDS:
                ref_cmds_local.append(c)

    local_config = WorkerConfig(
        envs=config.envs,
        include_forward=config.include_forward,
        emit_forward=config.emit_forward,
        lowercase_labels=config.lowercase_labels,
        label_map=config.label_map,
        ref_commands=ref_cmds_local,
        fuzzy=config.fuzzy,
        fuzzy_threshold=config.fuzzy_threshold,
        include_statement_refs=config.include_statement_refs,
        ordered_output=config.ordered_output,
    )

    env_to_kind = infer_env_kind_map_from_newtheorem(all_text)
    inferred_envs = sorted(env_to_kind.keys())
    envs_merged = merge_envs(local_config.envs, inferred_envs)

    events = scan_events(cleaned_texts, envs_merged)
    stmt_nodes, key_to_idx = pass1_collect_nodes(events, env_to_kind)

    # CHANGED: build_edges now also returns proof_nodes
    edges, fwd_edges, unknown_refs, skipped_forward_count, proof_nodes = build_edges(
        events=events,
        nodes=stmt_nodes,
        key_to_idx=key_to_idx,
        config=local_config
    )

    skipped_forward = [skipped_forward_count]
    # NOTE: statement-refs stays statement-only (does not scan proof nodes)
    add_statement_edges(
        nodes=stmt_nodes,
        edges=edges,
        forward_edges=fwd_edges,
        unknown_refs=unknown_refs,
        config=local_config,
        skipped_forward=skipped_forward
    )

    edges = dedupe_edges(edges)
    if local_config.emit_forward:
        fwd_edges = dedupe_edges(fwd_edges)
    else:
        fwd_edges = []

    # NEW: include proof nodes in the final graph
    nodes_all = stmt_nodes + proof_nodes

    ids = [n.id for n in nodes_all]
    topo, dag_valid, adjacency = topological_sort_ids(ids, edges)

    out = {
        "filenames": filenames,
        "nodes": [
            {
                "id": n.id,
                "env": n.env,
                "env_raw": n.env_raw,
                "label": n.label,
                "file": n.file,
                "ordinal": n.ordinal,
                "text": n.text,
            }
            for n in nodes_all
        ],
        "edges": [{"from": u, "to": v} for (u, v) in edges],
        "adjacency": adjacency,
        "topo_order": topo,
        "dag_valid": dag_valid,
        "unknown_refs": unknown_refs,
        "skipped_forward": skipped_forward[0],
    }

    if local_config.emit_forward:
        out["forward_edges"] = [{"from": u, "to": v} for (u, v) in fwd_edges]

    return out

def format_error_line(index: int, msg: str, filenames: Optional[List[str]] = None) -> str:
    obj: Dict = {"index": index, "error": msg}
    if filenames is not None:
        obj["filenames"] = filenames
    return json.dumps(obj, ensure_ascii=False)

def process_json_line(index: int, raw_line: str, config: WorkerConfig) -> Tuple[str, bool]:
    try:
        rec = json.loads(raw_line)
    except Exception as e:
        return format_error_line(index, f"JSON parse error: {e}"), True

    filenames = rec.get("filenames")
    latex_list = rec.get("latex")

    if not isinstance(filenames, list) or not isinstance(latex_list, list):
        return format_error_line(index, "Record must contain 'filenames' (list) and 'latex' (list)."), True

    try:
        out = process_record(filenames, latex_list, config)
        out["index"] = index
        return json.dumps(out, ensure_ascii=False), False
    except Exception as e:
        return format_error_line(index, f"Processing error: {e}", filenames=filenames), True


# ---------------------------
# Writer + workers (multiprocessing)
# ---------------------------

def writer_loop(
    result_queue: mp.Queue,
    tmp_output_path: str,
    ordered_output: bool,
    written_counter: mp.Value
) -> None:
    buffer: Dict[int, str] = {}
    next_index = 0

    with open(tmp_output_path, "w", encoding="utf-8") as fout:
        while True:
            item = result_queue.get()
            if item is None:
                break

            idx, line = item

            if not ordered_output:
                fout.write(line + "\n")
                with written_counter.get_lock():
                    written_counter.value += 1
                continue

            buffer[idx] = line
            while next_index in buffer:
                fout.write(buffer.pop(next_index) + "\n")
                with written_counter.get_lock():
                    written_counter.value += 1
                next_index += 1

        if ordered_output and buffer:
            for idx in sorted(buffer):
                fout.write(buffer[idx] + "\n")
                with written_counter.get_lock():
                    written_counter.value += 1

def worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config: WorkerConfig,
    err_counter: mp.Value,
    err_lock: mp.Lock
) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            break
        idx, raw_line = task
        out_line, is_error = process_json_line(idx, raw_line, config)
        result_queue.put((idx, out_line))
        if is_error:
            with err_lock:
                err_counter.value += 1

def ensure_readable_file(path: str, desc: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")

def load_label_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    ensure_readable_file(path, "Label map")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--label-map must be a JSON object mapping alias -> canonical label")
    return data

def _count_records_in_jsonl(input_path: str) -> int:
    total = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                total += 1
    return total

def run_parallel_jsonl(
    input_path: str,
    output_path: str,
    config: WorkerConfig,
    workers: int,
    queue_maxsize: int
) -> Tuple[int, int]:
    ensure_readable_file(input_path, "Input JSONL")

    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_output_path = output_path + ".tmp"

    total_records = _count_records_in_jsonl(input_path)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=queue_maxsize)
    result_queue = ctx.Queue(maxsize=queue_maxsize)

    err_counter = ctx.Value("i", 0)
    err_lock = ctx.Lock()

    written_counter = ctx.Value("i", 0)

    writer = ctx.Process(target=writer_loop, args=(result_queue, tmp_output_path, config.ordered_output, written_counter))
    writer.start()

    procs: List[mp.Process] = []
    for _ in range(max(1, workers)):
        p = ctx.Process(target=worker_loop, args=(task_queue, result_queue, config, err_counter, err_lock))
        p.start()
        procs.append(p)

    tqdm_cls = None
    try:
        from tqdm import tqdm  # type: ignore
        tqdm_cls = tqdm
    except Exception:
        tqdm_cls = None

    # Queueing progress
    record_index = 0
    if tqdm_cls is not None:
        pbar_q = tqdm_cls(total=total_records, unit="rec", desc="queued", leave=True)
    else:
        pbar_q = None

    last_print_t = time.time()
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            task_queue.put((record_index, line.rstrip("\n")))
            record_index += 1
            if pbar_q is not None:
                pbar_q.update(1)
            else:
                now = time.time()
                if now - last_print_t >= 1.0:
                    print(f"[progress] queued {record_index}/{total_records} record(s)", file=sys.stderr)
                    last_print_t = now

    if pbar_q is not None:
        pbar_q.close()

    for _ in procs:
        task_queue.put(None)

    # Writing progress
    if tqdm_cls is not None:
        pbar_w = tqdm_cls(total=record_index, unit="rec", desc="written", leave=True)
    else:
        pbar_w = None

    ### CHANGED: avoid deadlock by not waiting for writer to exit before sending sentinel
    last_val = 0
    last_print_t = time.time()

    # Phase 1: wait for workers to finish, keep updating progress
    while any(p.is_alive() for p in procs):
        with written_counter.get_lock():
            cur = int(written_counter.value)
        if pbar_w is not None:
            if cur > last_val:
                pbar_w.update(cur - last_val)
                last_val = cur
        else:
            now = time.time()
            if now - last_print_t >= 1.0:
                print(f"[progress] wrote {cur}/{record_index} record(s)", file=sys.stderr)
                last_print_t = now
        time.sleep(0.2)

    for p in procs:
        p.join()

    # Phase 2: now tell writer to stop, then wait for writer to exit
    result_queue.put(None)

    while writer.is_alive():
        with written_counter.get_lock():
            cur = int(written_counter.value)
        if pbar_w is not None:
            if cur > last_val:
                pbar_w.update(cur - last_val)
                last_val = cur
        else:
            now = time.time()
            if now - last_print_t >= 1.0:
                print(f"[progress] wrote {cur}/{record_index} record(s)", file=sys.stderr)
                last_print_t = now
        time.sleep(0.2)

    writer.join()
    ### END CHANGED

    # Final tqdm flush
    if pbar_w is not None:
        with written_counter.get_lock():
            cur = int(written_counter.value)
        if cur > last_val:
            pbar_w.update(cur - last_val)
        pbar_w.close()

    if writer.exitcode != 0:
        raise RuntimeError(f"Writer process exited with code {writer.exitcode}")

    os.replace(tmp_output_path, output_path)
    return record_index, int(err_counter.value)


# ---------------------------
# Stats computation from output JSONL
# ---------------------------

def aggregate_stats_from_output(output_path: str) -> Dict:
    stats = {
        "records": 0,
        "error_records": 0,
        "total_nodes": 0,
        "total_edges": 0,
        "total_forward_edges": 0,
        "total_unknown_ref_labels": 0,
        "total_unknown_ref_uses": 0,
        "total_skipped_forward": 0,
        "edge_types": {},
    }

    ensure_readable_file(output_path, "Output JSONL")

    with open(output_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                stats["error_records"] += 1
                continue

            if "error" in obj:
                stats["error_records"] += 1
                continue

            stats["records"] += 1

            nodes = obj.get("nodes", [])
            edges = obj.get("edges", [])
            fwd_edges = obj.get("forward_edges", [])
            unknown_refs = obj.get("unknown_refs", {})
            skipped_forward = obj.get("skipped_forward", 0)

            stats["total_nodes"] += len(nodes)
            stats["total_edges"] += len(edges)
            stats["total_forward_edges"] += len(fwd_edges)
            stats["total_unknown_ref_labels"] += len(unknown_refs)
            stats["total_unknown_ref_uses"] += sum(unknown_refs.values())
            stats["total_skipped_forward"] += skipped_forward

            id_to_env = {n.get("id"): n.get("env", "?") for n in nodes}
            for e in edges:
                src = e.get("from")
                dst = e.get("to")
                src_env = id_to_env.get(src, "?")
                dst_env = id_to_env.get(dst, "?")
                key = (str(src_env), str(dst_env))
                stats["edge_types"][key] = stats["edge_types"].get(key, 0) + 1

    return stats

def print_stats(stats: Dict, num_records: int, num_errors: int) -> None:
    print("\n[stats] ---------- DAG Summary ----------")
    print(f"[stats] Input records:       {num_records}")
    print(f"[stats] Error records:       {num_errors}")
    print(f"[stats] Successful records:  {max(0, num_records - num_errors)}")
    print(f"[stats] Total nodes:         {stats['total_nodes']}")
    print(f"[stats] Total edges:         {stats['total_edges']}")
    print(f"[stats] Total forward edges: {stats['total_forward_edges']}")
    print(f"[stats] Total skipped_forward (not turned into edges): {stats['total_skipped_forward']}")
    print(f"[stats] Unknown ref labels:  {stats['total_unknown_ref_labels']}")
    print(f"[stats] Unknown ref uses:    {stats['total_unknown_ref_uses']}")

    if stats["edge_types"]:
        print("[stats] Edge types by (src_env -> dst_env):")
        items = sorted(
            stats["edge_types"].items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )
        for (src_env, dst_env), count in items:
            print(f"[stats]   {src_env} -> {dst_env}: {count}")
    else:
        print("[stats] Edge types by (src_env -> dst_env): (none)")

    print("[stats] ----------------------------------\n")


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dependency DAGs from JSONL of LaTeX sources (multiprocessing).")
    p.add_argument("-i", "--input", required=True, help="Path to input JSONL")
    p.add_argument("-o", "--output", required=True, help="Path to output JSONL")

    p.add_argument("--envs", default=",".join(DEFAULT_ENV_NAMES), help="Comma-separated environments to scan for nodes")
    p.add_argument("--include-forward", action="store_true", help="Include edges to forward references (may create cycles)")
    p.add_argument("--emit-forward", action="store_true", help="Also emit forward edges under 'forward_edges'")
    p.add_argument("--label-map", help="Path to JSON file mapping alias -> canonical labels to remap refs")
    p.add_argument("--lowercase-labels", action="store_true", help="Match labels case-insensitively by lowercasing")
    p.add_argument("--refcmds", default=",".join(DEFAULT_REFCMDS), help="Comma-separated ref-like commands to recognize")
    p.add_argument("--fuzzy-labels", action="store_true", help="Enable fuzzy matching for ref labels")
    p.add_argument("--fuzzy-threshold", type=float, default=0.66, help="Threshold (0-1) for token-set fuzzy matching")
    p.add_argument("--include-statement-refs", action="store_true", help="Also treat refs inside statements as dependencies")

    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)), help="Number of worker processes")
    p.add_argument("--queue-maxsize", type=int, default=128, help="Max queue size for tasks/results")
    p.add_argument("--unordered-output", action="store_true", help="Do not preserve record order in output")

    return p.parse_args(argv if argv is not None else sys.argv[1:])

def build_config_from_args(args: argparse.Namespace) -> WorkerConfig:
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    refcmds = [c.strip() for c in args.refcmds.split(",") if c.strip()]
    label_map = load_label_map(args.label_map)

    return WorkerConfig(
        envs=envs,
        include_forward=args.include_forward,
        emit_forward=args.emit_forward,
        lowercase_labels=args.lowercase_labels,
        label_map=label_map,
        ref_commands=refcmds,
        fuzzy=args.fuzzy_labels,
        fuzzy_threshold=float(args.fuzzy_threshold),
        include_statement_refs=args.include_statement_refs,
        ordered_output=(not args.unordered_output),
    )

def main() -> None:
    args = parse_args()
    ensure_readable_file(args.input, "Input JSONL")
    _ = load_label_map(args.label_map)

    config = build_config_from_args(args)

    num_records, num_errors = run_parallel_jsonl(
        input_path=args.input,
        output_path=args.output,
        config=config,
        workers=max(1, int(args.workers)),
        queue_maxsize=max(1, int(args.queue_maxsize))
    )

    print(f"[ok] Processed {num_records} record(s). Wrote: {args.output}")
    if num_errors:
        print(f"[warn] {num_errors} record(s) had errors; see lines with an 'error' field in {args.output}.")

    stats = aggregate_stats_from_output(args.output)
    print_stats(stats, num_records, num_errors)

if __name__ == "__main__":
    main()
