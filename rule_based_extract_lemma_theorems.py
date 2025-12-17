#!/usr/bin/env python3
r"""
rule_based_extract_lemma_theorems.py

JSONL in (each line has a list of LaTeX source strings) → JSONL out (one line per processed input that had any extracted environments; with multiple workers, each writes only its assigned non-empty lines):
    {
      "definitions": [{"index": int, "text": str}],
      "propositions": [{"index": int, "text": str}],
      "examples": [{"index": int, "text": str}],
      "remarks": [{"index": int, "text": str}],
      "proofs": [{"index": int, "text": str}],
      "corollaries": [{"index": int, "text": str}],
      "assumptions": [{"index": int, "text": str}],
      "theorems": [{"index": int, "text": str}],
      "lemmas": [{"index": int, "text": str}],
      // passthrough fields from input if present
      "path": str
    }

Enhancements:
- [FIX] Case-insensitive environment detection everywhere.
- [FALLBACK] Regex-based extraction for common theorem-like blocks (incl. stars).
- [DEDUP] Stable de-duplication across primary and fallback passes.
- [FIELD] --field option (default: 'latex'), plus auto-detect if missing.
- [DEBUG] --debug prints per-line counts and a tiny env summary.
- [STATS] Prints aggregate stats at end.

Usage:
    python rule_based_extract_lemma_theorems.py --in papers.jsonl --out statements.jsonl

Parallelization:
- Use --num_workers N to auto-spawn N local workers. Each worker processes
  lines where (zero_based_index % N == worker_id). Outputs are merged into the
  final --out file. To run single-process, keep --num_workers at 1.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import os
import multiprocessing as mp
from typing import Dict, List, Optional, Set, Tuple

# =========================
# Configuration / Aliases
# =========================

TARGET_TYPES: Set[str] = {
    "theorem", "lemma",
    "definition", "proposition", "example", "remark", "proof", "corollary", "assumption",
}

# Map canonical type -> plural result key
RESULT_KEYS: Dict[str, str] = {
    "theorem": "theorems",
    "lemma": "lemmas",
    "definition": "definitions",
    "proposition": "propositions",
    "example": "examples",
    "remark": "remarks",
    "proof": "proofs",
    "corollary": "corollaries",
    "assumption": "assumptions",
}

COMMON_ENV_ALIASES: Dict[str, str] = {
    # theorems & lemmas
    "thm": "theorem", "thma": "theorem", "thmb": "theorem",
    "lem": "lemma",   "lema": "lemma",   "lemb": "lemma",
    # definitions
    "def": "definition", "defn": "definition", "dfn": "definition",
    # propositions
    "prop": "proposition", "propo": "proposition",
    # examples
    "ex": "example", "exa": "example", "exmp": "example",
    # remarks
    "rmk": "remark", "rem": "remark",
    # corollaries
    "cor": "corollary", "corol": "corollary",
    # assumptions
    "assump": "assumption", "assum": "assumption",
    # proofs
    "pf": "proof",
}

STRIP_WHOLE_ENVS: Set[str] = {
    "comment", "verbatim", "verbatim*", "verbatimtab", "Verbatim",
    "lstlisting", "minted", "alltt"
}

# =========================
# I/O Safety Helpers
# =========================

def ensure_input_exists(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"--in does not exist or is not a file: {path}")

def ensure_output_path(out_path: Path, force: bool) -> None:
    parent = out_path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {out_path}\nUse --force to overwrite."
        )

# =========================
# Light-weight LaTeX utils
# =========================

def remove_inline_comments(text: str) -> str:
    r"""Remove unescaped % comments line-by-line."""
    out_lines: List[str] = []
    for line in text.splitlines():
        i = 0
        cut = len(line)
        while i < len(line):
            if line[i] == "%":
                if i == 0 or line[i - 1] != "\\":
                    cut = i
                    break
            i += 1
        out_lines.append(line[:cut])
    return "\n".join(out_lines)

def _compile_env_block_regex(env: str) -> re.Pattern:
    # Case-insensitive so it removes \begin{Verbatim} and \begin{verbatim} equally.
    pattern = (
        r"\\begin\{" + re.escape(env) + r"\}"
        r"(?:\s*\[[^\]]*\])?"       # optional [...]
        r"(?:\s*\{[^}]*\})?"        # optional {..}
        r"(?:\s*\{[^}]*\})?"
        r"(?:\s*\{[^}]*\})?"
        r"(.*?)"
        r"\\end\{" + re.escape(env) + r"\}"
    )
    return re.compile(pattern, re.DOTALL | re.IGNORECASE)

def strip_env_blocks(text: str, envs: Set[str]) -> str:
    r"""Remove entire blocks of specified envs to avoid stray \begin/\end noise."""
    changed = True
    while changed:
        changed = False
        for env in envs:
            rx = _compile_env_block_regex(env)
            new_text, n = rx.subn("", text)
            if n > 0:
                text = new_text
                changed = True
    return text

def parse_bracketed_group(text: str, pos: int, open_ch: str, close_ch: str) -> Tuple[Optional[str], int]:
    r"""Parse a balanced bracketed group starting at pos."""
    n = len(text)
    if pos >= n or text[pos] != open_ch:
        return None, pos
    depth = 0
    i = pos
    while i < n:
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[pos + 1:i], i + 1
        elif ch == "\\" and i + 1 < n:
            i += 1  # skip escaped char
        i += 1
    return None, pos  # unbalanced

def skip_spaces(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos].isspace():
        pos += 1
    return pos

def find_matching_end(text: str, env: str, start_pos: int) -> Optional[re.Match]:
    r"""Find matching \end{env} balancing nested {env}, case-insensitive."""
    token = re.compile(r"\\(begin|end)\{" + re.escape(env) + r"\}", re.DOTALL | re.IGNORECASE)
    depth = 1
    for m in token.finditer(text, start_pos):
        if m.group(1).lower() == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m
    return None

# =========================
# Theorem-env discovery
# =========================

def parse_newtheorem_defs(text: str, env_display: Dict[str, Optional[str]]) -> None:
    r"""Parse \newtheorem definitions and collect env display names."""
    rx = re.compile(
        r"\\newtheorem\*?\s*"
        r"\{\s*([^\}]+)\s*\}"
        r"(?:\s*\[\s*([^\]]+)\s*\])?"
        r"\s*\{\s*([^\}]+)\s*\}"
        r"(?:\s*\[\s*([^\]]+)\s*\])?",
        re.IGNORECASE
    )
    for m in rx.finditer(text):
        name = (m.group(1) or "").strip()
        printed = (m.group(3) or "").strip()
        if name:
            env_display.setdefault(name, printed if printed else None)

def parse_spnewtheorem_defs(text: str, env_display: Dict[str, Optional[str]]) -> None:
    r"""Parse \spnewtheorem definitions and collect env display names."""
    rx = re.compile(
        r"\\spnewtheorem\*?\s*"
        r"\{\s*([^\}]+)\s*\}\s*"
        r"\{\s*([^\}]+)\s*\}",
        re.IGNORECASE
    )
    for m in rx.finditer(text):
        name = (m.group(1) or "").strip()
        printed = (m.group(2) or "").strip()
        if name:
            env_display.setdefault(name, printed if printed else None)

def parse_declaretheorem_defs(text: str, env_display: Dict[str, Optional[str]]) -> None:
    r"""Parse \declaretheorem[options]{name} and extract name=Display if present."""
    rx = re.compile(
        r"\\declaretheorem(?:\s*\[([^\]]*)\])?\s*\{\s*([^\}]+)\s*\}",
        re.IGNORECASE | re.DOTALL
    )
    for m in rx.finditer(text):
        opts = (m.group(1) or "")
        name = (m.group(2) or "").strip()
        disp = None
        for piece in opts.split(","):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            k, v = piece.split("=", 1)
            if k.strip() == "name":
                disp = v.strip()
                if len(disp) >= 2 and ((disp[0] == "{" and disp[-1] == "}") or (disp[0] == "[" and disp[-1] == "]")):
                    disp = disp[1:-1].strip()
                break
        if name:
            env_display.setdefault(name, disp if disp else None)

def parse_newaliascnt_defs(text: str, env_display: Dict[str, Optional[str]]) -> None:
    r"""Parse \newaliascnt{new}{existing}; record 'new' as an env."""
    rx = re.compile(
        r"\\newaliascnt\s*\{\s*([^\}]+)\s*\}\s*\{\s*([^\}]+)\s*\}",
        re.IGNORECASE
    )
    for m in rx.finditer(text):
        new = (m.group(1) or "").strip()
        existing = (m.group(2) or "").strip()
        if not new or not existing:
            continue
        if existing in env_display and env_display[existing]:
            env_display.setdefault(new, env_display[existing])
        else:
            env_display.setdefault(new, None)

def collect_env_definitions(all_texts: List[str]) -> Dict[str, Optional[str]]:
    r"""Aggregate theorem-like environment display names across all sources."""
    env_display: Dict[str, Optional[str]] = {}
    for text in all_texts:
        t = strip_env_blocks(text, STRIP_WHOLE_ENVS)
        t = remove_inline_comments(t)
        parse_newtheorem_defs(t, env_display)
        parse_spnewtheorem_defs(t, env_display)
        parse_declaretheorem_defs(t, env_display)
        parse_newaliascnt_defs(t, env_display)
    return env_display

# =========================
# Extraction helpers
# =========================

def _display_to_canonical(disp: Optional[str]) -> Optional[str]:
    r"""Map a printed display name to 'theorem' or 'lemma' if recognizable."""
    if not disp:
        return None
    dl = disp.lower()
    dl = re.sub(r"[^a-z ]+", " ", dl)
    dl = " ".join(dl.split())
    if "theorem" in dl:
        return "theorem"
    if "lemma" in dl:
        return "lemma"
    if "definition" in dl:
        return "definition"
    if "proposition" in dl:
        return "proposition"
    if "example" in dl:
        return "example"
    if "remark" in dl:
        return "remark"
    if "proof" in dl:
        return "proof"
    if "corollary" in dl:
        return "corollary"
    if "assumption" in dl:
        return "assumption"
    return None

def canonicalize_env_name(env: str, env_display: Dict[str, Optional[str]]) -> Tuple[str, bool]:
    r"""Return (canonical_type, starred) for an environment name, case-insensitive."""
    starred = env.endswith("*")
    base = env[:-1] if starred else env
    base_lc = base.strip().lower()

    # display-based resolution
    disp = env_display.get(base, None) or env_display.get(base_lc, None)
    cand = _display_to_canonical(disp) if disp else None
    if cand:
        return cand, starred

    # alias-based
    if base_lc in COMMON_ENV_ALIASES:
        return COMMON_ENV_ALIASES[base_lc], starred

    if base_lc in TARGET_TYPES:
        return base_lc, starred
    if base_lc.startswith("theorem"):
        return "theorem", starred
    if base_lc.startswith("lemma"):
        return "lemma", starred
    if base_lc.startswith("definition"):
        return "definition", starred
    if base_lc.startswith("proposition") or base_lc.startswith("prop"):
        return "proposition", starred
    if base_lc.startswith("example") or base_lc in {"ex", "exa", "exmp"}:
        return "example", starred
    if base_lc.startswith("remark") or base_lc in {"rmk", "rem"}:
        return "remark", starred
    if base_lc.startswith("proof") or base_lc == "pf":
        return "proof", starred
    if base_lc.startswith("corollary") or base_lc.startswith("corol") or base_lc == "cor":
        return "corollary", starred
    if base_lc.startswith("assumption") or base_lc.startswith("assump") or base_lc.startswith("assum"):
        return "assumption", starred

    return base_lc, starred

def extract_title_after_begin(text: str, pos_after_begin: int, env: str) -> Tuple[Optional[str], int, Optional[str], Optional[str]]:
    r"""
    Parse tokens immediately after \begin{env}:
      - general: optional [Title]
      - restatable: [opts]? {inner_env} {macro} [Title]?
    Return: (title, new_pos, restatable_inner_env, restatable_macro)
    """
    pos = skip_spaces(text, pos_after_begin)
    title = None
    rest_inner = None
    rest_macro = None

    if env.lower() == "restatable":
        if pos < len(text) and text[pos] == "[":
            _, pos = parse_bracketed_group(text, pos, "[", "]")
            pos = skip_spaces(text, pos)
        inner, pos2 = parse_bracketed_group(text, pos, "{", "}")
        if inner:
            rest_inner = inner.strip()
            pos = skip_spaces(text, pos2)
        macro, pos3 = parse_bracketed_group(text, pos, "{", "}")
        if macro:
            rest_macro = macro.strip()
            pos = skip_spaces(text, pos3)
        if pos < len(text) and text[pos] == "[":
            t, pos = parse_bracketed_group(text, pos, "[", "]")
            if t is not None:
                title = t.strip()
        return title, pos, rest_inner, rest_macro

    if pos < len(text) and text[pos] == "[":
        t, pos = parse_bracketed_group(text, pos, "[", "]")
        if t is not None:
            title = t.strip()
    return title, pos, None, None

def primary_extract_env_statements(text: str,
                                   env_display: Dict[str, Optional[str]],
                                   target_types: Set[str],
                                   debug: bool = False) -> Dict[str, List[str]]:
    r"""
    Primary balanced scanner (legacy): returns {type: [body, ...]}.
    """
    out: Dict[str, List[str]] = {t: [] for t in target_types}

    # Preprocess
    text2 = strip_env_blocks(text, STRIP_WHOLE_ENVS)
    text2 = remove_inline_comments(text2)

    begin_rx = re.compile(r"\\begin\{([^\}]+)\}", re.DOTALL | re.IGNORECASE)
    seen_envs = []

    for m in begin_rx.finditer(text2):
        env_raw = m.group(1).strip()
        seen_envs.append(env_raw)
        begin_token_end = m.end()

        _title, hdr_end_pos, rest_inner, _rest_macro = extract_title_after_begin(text2, begin_token_end, env_raw)
        env_to_use = rest_inner if (env_raw.lower() == "restatable" and rest_inner) else env_raw

        end_match = find_matching_end(text2, env_raw, hdr_end_pos)
        if not end_match:
            continue  # unbalanced or malformed

        body = text2[hdr_end_pos:end_match.start()]
        canonical_type, _starred = canonicalize_env_name(env_to_use, env_display)

        if canonical_type in target_types:
            cleaned = re.sub(r"\\label\{[^}]*\}", "", body).strip()
            if cleaned:
                out[canonical_type].append(cleaned)

    if debug:
        # Print only a short sample of env names to avoid clutter
        sample = ", ".join(seen_envs[:10])
        print(f"[DEBUG] Seen envs: {sample}{' ...' if len(seen_envs) > 10 else ''}", file=sys.stderr)

    return out

def primary_scan_env_events(text: str,
                            env_display: Dict[str, Optional[str]],
                            target_types: Set[str],
                            debug: bool = False) -> List[Tuple[str, str, int]]:
    r"""
    Primary balanced scanner that returns a list of (type, body, begin_pos) events.
    begin_pos is the index of the \begin token in a lightly preprocessed text.
    """
    events: List[Tuple[str, str, int]] = []

    text2 = strip_env_blocks(text, STRIP_WHOLE_ENVS)
    text2 = remove_inline_comments(text2)

    begin_rx = re.compile(r"\\begin\{([^\}]+)\}", re.DOTALL | re.IGNORECASE)
    seen_envs = []
    for m in begin_rx.finditer(text2):
        env_raw = m.group(1).strip()
        seen_envs.append(env_raw)
        begin_token_end = m.end()

        _title, hdr_end_pos, rest_inner, _rest_macro = extract_title_after_begin(text2, begin_token_end, env_raw)
        env_to_use = rest_inner if (env_raw.lower() == "restatable" and rest_inner) else env_raw

        end_match = find_matching_end(text2, env_raw, hdr_end_pos)
        if not end_match:
            continue

        body = text2[hdr_end_pos:end_match.start()]
        canonical_type, _starred = canonicalize_env_name(env_to_use, env_display)
        if canonical_type in target_types:
            cleaned = re.sub(r"\\label\{[^}]*\}", "", body).strip()
            if cleaned:
                events.append((canonical_type, cleaned, m.start()))

    if debug:
        sample = ", ".join(seen_envs[:10])
        print(f"[DEBUG] Seen envs: {sample}{' ...' if len(seen_envs) > 10 else ''}", file=sys.stderr)
    return events

# --------- FALLBACK: direct regex blocks (robust but less precise) ---------

FALLBACK_RX = re.compile(
    r"\\begin\{(?P<env>(?:lemma|theorem|definition|proposition|example|remark|proof|corollary|assumption)\*?)\}"
    r"(?:\s*\[[^\]]*\])?"
    r"(?P<body>.*?)"
    r"\\end\{(?P=env)\}",
    re.IGNORECASE | re.DOTALL
)

def fallback_extract_env_statements(text: str) -> Dict[str, List[str]]:
    r"""Regex-only fallback for common theorem-like blocks (incl. stars)."""
    out: Dict[str, List[str]] = {t: [] for t in TARGET_TYPES}

    text2 = remove_inline_comments(strip_env_blocks(text, STRIP_WHOLE_ENVS))

    for m in FALLBACK_RX.finditer(text2):
        env = m.group("env").strip().lower()
        body = m.group("body")
        body = re.sub(r"\\label\{[^}]*\}", "", body).strip()
        if not body:
            continue
        if env.startswith("lemma"):
            out["lemma"].append(body)
        elif env.startswith("theorem"):
            out["theorem"].append(body)
        elif env.startswith("definition"):
            out["definition"].append(body)
        elif env.startswith("proposition"):
            out["proposition"].append(body)
        elif env.startswith("example"):
            out["example"].append(body)
        elif env.startswith("remark"):
            out["remark"].append(body)
        elif env.startswith("proof"):
            out["proof"].append(body)
        elif env.startswith("corollary"):
            out["corollary"].append(body)
        elif env.startswith("assumption"):
            out["assumption"].append(body)
    return out

def fallback_scan_env_events(text: str) -> List[Tuple[str, str, int]]:
    r"""Regex-only fallback that returns a list of (type, body, begin_pos) events."""
    events: List[Tuple[str, str, int]] = []
    text2 = remove_inline_comments(strip_env_blocks(text, STRIP_WHOLE_ENVS))
    for m in FALLBACK_RX.finditer(text2):
        env = m.group("env").strip().lower()
        body = re.sub(r"\\label\{[^}]*\}", "", m.group("body")).strip()
        if not body:
            continue
        if env.startswith("lemma"):
            t = "lemma"
        elif env.startswith("theorem"):
            t = "theorem"
        elif env.startswith("definition"):
            t = "definition"
        elif env.startswith("proposition"):
            t = "proposition"
        elif env.startswith("example"):
            t = "example"
        elif env.startswith("remark"):
            t = "remark"
        elif env.startswith("proof"):
            t = "proof"
        elif env.startswith("corollary"):
            t = "corollary"
        elif env.startswith("assumption"):
            t = "assumption"
        else:
            continue
        events.append((t, body, m.start()))
    return events

# =========================
# Per-line processing
# =========================

def _stable_extend_dedup(dst: List[str], src: List[str], seen: Set[str]) -> None:
    r"""Append from src to dst if not seen, preserving order."""
    for s in src:
        if s not in seen:
            dst.append(s)
            seen.add(s)

def get_latex_list_from_obj(obj: dict, preferred_field: str, debug: bool = False) -> List[str]:
    r"""
    Return a list[str] of LaTeX sources from obj.
    Priority:
      1) obj[preferred_field] if it is a list[str]
      2) auto-detect: first key whose value is a list[str] and contains '\begin{' or '\documentclass'
    """
    val = obj.get(preferred_field)
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, list) and all(isinstance(x, str) for x in val):
        return [s for s in val if s]

    # auto-detect
    best_key = None
    best_score = -1
    for k, v in obj.items():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            joined = "\n".join(v)
        elif isinstance(v, str):
            joined = v
        else:
            joined = None
        if joined is None:
            continue
        score = 0
        if "\\begin{" in joined:
            score += 2
        if "\\documentclass" in joined:
            score += 1
        if score > best_score:
            best_score = score
            best_key = k
    if best_key and debug:
        print(f"[DEBUG] Auto-detected field '{best_key}' for LaTeX list", file=sys.stderr)
    if best_key:
        v = obj[best_key]
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return [s for s in v if s]
        return []

    return []

def process_jsonl_line(obj: dict, preferred_field: str, debug: bool = False) -> Dict[str, List[Dict[str, object]]]:
    r"""
    Return a dict with lists for: definitions, propositions, examples, remarks,
    proofs, corollaries, assumptions, theorems, lemmas. Each list contains
    items with fields {"index": int, "text": str}, where index is a unique
    global order across all extracted environments within this record.
    """
    texts: List[str] = get_latex_list_from_obj(obj, preferred_field, debug=debug)
    if not texts:
        return {rk: [] for rk in RESULT_KEYS.values()}

    env_display = collect_env_definitions(texts)

    # Collect events with positions from primary and fallback passes
    # Keep the earliest occurrence if duplicates (by type+text)
    best: Dict[Tuple[str, str], Tuple[int, int]] = {}
    # key -> (txt_idx, local_pos)

    for ti, s in enumerate(texts):
        for t, body, pos in primary_scan_env_events(s, env_display, TARGET_TYPES, debug=debug):
            key = (t, body)
            cur = best.get(key)
            if cur is None or (ti, pos) < cur:
                best[key] = (ti, pos)

    for ti, s in enumerate(texts):
        for t, body, pos in fallback_scan_env_events(s):
            key = (t, body)
            cur = best.get(key)
            if cur is None or (ti, pos) < cur:
                best[key] = (ti, pos)

    # Sort all unique events by (txt_idx, local_pos)
    ordered = sorted(((t, body, loc[0], loc[1]) for (t, body), loc in best.items()), key=lambda x: (x[2], x[3]))

    # Assign global indices and materialize per-type lists
    out: Dict[str, List[Dict[str, object]]] = {rk: [] for rk in RESULT_KEYS.values()}
    for idx, (t, body, _ti, _pos) in enumerate(ordered):
        out_key = RESULT_KEYS.get(t, t + "s")
        out[out_key].append({"index": idx, "text": body})

    return out

# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract theorem-like statements from a JSONL file of LaTeX sources.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input JSONL file.")
    parser.add_argument("--out", dest="out_path", required=True, help="Output JSONL file (one object per input line).")
    parser.add_argument("--field", default="latex", help="JSON key containing the list[str] of LaTeX sources (default: 'latex').")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument("--debug", action="store_true", help="Print per-line debug info to stderr.")
    parser.add_argument("--num_workers", type=int, default=1, help="Total local workers to auto-spawn (default: 1).")
    args = parser.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()

    try:
        ensure_input_exists(in_path)
        ensure_output_path(out_path, force=args.force)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Validate settings
    if args.num_workers < 1:
        print("[ERROR] --num_workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    def process_shard(in_path: Path, out_path: Path, preferred_field: str, debug: bool, num_workers: int, worker_id: int) -> Dict[str, object]:
        total_papers = 0           # processed lines for this shard
        total_statements = 0
        papers_with_any = 0        # lines that produced any extracted statements (written)
        env_totals: Dict[str, int] = {rk: 0 for rk in RESULT_KEYS.values()}
        env_papers_with_any: Dict[str, int] = {rk: 0 for rk in RESULT_KEYS.values()}

        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for idx0, line in enumerate(fin):
                if (idx0 % num_workers) != worker_id:
                    continue
                line_num = idx0 + 1
                s = line.strip()
                if not s:
                    result = {rk: [] for rk in RESULT_KEYS.values()}
                else:
                    try:
                        obj = json.loads(s)
                        result = process_jsonl_line(obj, preferred_field, debug=debug)
                        # passthrough selected input fields
                        if isinstance(obj, dict) and "path" in obj:
                            result["path"] = obj["path"]
                    except Exception as e:
                        print(f"[WARN] Worker {worker_id} line {line_num}: JSON decode error: {e}", file=sys.stderr)
                        result = {rk: [] for rk in RESULT_KEYS.values()}

                # Count all extracted statements across known result keys
                all_keys = list(RESULT_KEYS.values())
                n_here = sum(len(result.get(k, [])) for k in all_keys)
                if debug:
                    counts = ", ".join(f"{k}={len(result.get(k, []))}" for k in all_keys)
                    print(f"[DEBUG] Worker {worker_id} line {line_num}: {counts}", file=sys.stderr)

                # Always count a processed paper, even if empty; skip writing if empty
                total_papers += 1
                if (total_papers % 10000) == 0:
                    print(f"[INFO] Worker {worker_id}: {total_papers // 1000}K done", file=sys.stderr)
                if n_here == 0:
                    continue

                # Aggregate stats for non-empty records
                total_statements += n_here
                papers_with_any += 1
                for k in RESULT_KEYS.values():
                    n_k = len(result.get(k, []))
                    env_totals[k] += n_k
                    if n_k > 0:
                        env_papers_with_any[k] += 1

                # Write only non-empty records
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        return {
            "processed": total_papers,
            "written": papers_with_any,
            "total_statements": total_statements,
            "env_totals": env_totals,
            "env_papers_with_any": env_papers_with_any,
        }

    # Single-process path
    if args.num_workers == 1:
        stats = process_shard(in_path, out_path, args.field, args.debug, 1, 0)
        processed = int(stats["processed"]) or 0
        written = int(stats["written"]) or 0
        total_statements = int(stats["total_statements"]) or 0
        avg = (total_statements / processed) if processed else 0.0
        env_totals = stats["env_totals"]
        env_papers_with_any = stats["env_papers_with_any"]
        print(f"[OK] Processed {processed} lines; wrote {written} non-empty records to {out_path}")
        print(f"[STATS] Papers processed: {processed}")
        print(f"[STATS] Papers with any statements: {written}")
        print(f"[STATS] Total statements extracted: {total_statements}")
        print(f"[STATS] Average statements per paper: {avg:.3f}")
        per_type_totals = ", ".join(f"{k}={env_totals[k]}" for k in RESULT_KEYS.values())
        print(f"[STATS] Totals by type: {per_type_totals}")
        per_type_papers = ", ".join(f"{k}={env_papers_with_any[k]}" for k in RESULT_KEYS.values())
        print(f"[STATS] Papers with any by type: {per_type_papers}")
        return

    # Multi-process path: spawn N workers and merge outputs
    # Ensure final output path is available; part files live alongside it
    parent_dir = out_path.parent
    part_paths = [parent_dir / f"{out_path.name}.part-{i:04d}" for i in range(args.num_workers)]

    # Clean any pre-existing part files if forcing
    for p in part_paths:
        if p.exists() and not args.force:
            print(f"[ERROR] Part exists: {p}. Use --force or remove it.", file=sys.stderr)
            sys.exit(1)
        if p.exists() and args.force:
            try:
                os.remove(p)
            except OSError:
                pass

    print(f"[INFO] Spawning {args.num_workers} workers; merging into {out_path}", file=sys.stderr)

    def _worker_entry(i: int, q: mp.Queue) -> None:
        stats = process_shard(in_path, part_paths[i], args.field, args.debug, args.num_workers, i)
        q.put((i, stats))

    q: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []
    for i in range(args.num_workers):
        p = mp.Process(target=_worker_entry, args=(i, q), daemon=False)
        p.start()
        procs.append(p)

    # Collect stats from workers
    shard_stats: Dict[int, Dict[str, object]] = {}
    finished = 0
    while finished < args.num_workers:
        wid, stats = q.get()
        shard_stats[wid] = stats
        finished += 1

    # Join children
    for p in procs:
        p.join()

    # Merge parts into final out
    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(args.num_workers):
            part = part_paths[i]
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)

    # Aggregate stats
    processed = sum(int(s["processed"]) or 0 for s in shard_stats.values())
    written = sum(int(s["written"]) or 0 for s in shard_stats.values())
    total_statements = sum(int(s["total_statements"]) or 0 for s in shard_stats.values())
    avg = (total_statements / processed) if processed else 0.0
    env_totals: Dict[str, int] = {rk: 0 for rk in RESULT_KEYS.values()}
    env_papers_with_any: Dict[str, int] = {rk: 0 for rk in RESULT_KEYS.values()}
    for s in shard_stats.values():
        et = s["env_totals"]
        ep = s["env_papers_with_any"]
        for k in RESULT_KEYS.values():
            env_totals[k] += int(et[k])
            env_papers_with_any[k] += int(ep[k])

    print(f"[OK] Processed {processed} lines; wrote {written} non-empty records to {out_path}")
    print(f"[STATS] Papers processed: {processed}")
    print(f"[STATS] Papers with any statements: {written}")
    print(f"[STATS] Total statements extracted: {total_statements}")
    print(f"[STATS] Average statements per paper: {avg:.3f}")
    per_type_totals = ", ".join(f"{k}={env_totals[k]}" for k in RESULT_KEYS.values())
    print(f"[STATS] Totals by type: {per_type_totals}")
    per_type_papers = ", ".join(f"{k}={env_papers_with_any[k]}" for k in RESULT_KEYS.values())
    print(f"[STATS] Papers with any by type: {per_type_papers}")

if __name__ == "__main__":
    main()
