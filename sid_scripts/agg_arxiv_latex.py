#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

# =========================
# TeX detection helpers
# =========================

THEOREM_ENV_RE = re.compile(r"\\begin\s*\{\s*(theorem|thm)\s*\*?\s*\}", re.IGNORECASE)
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{\s*document\s*\}", re.IGNORECASE)
DOCUMENTCLASS_RE = re.compile(r"\\documentclass\b", re.IGNORECASE)
END_DOCUMENT_RE = re.compile(r"\\end\s*\{\s*document\s*\}", re.IGNORECASE)

INPUT_CMD_RE = re.compile(r"\\(input|include|subfile)\s*(\{[^}\n]+\}|[^\s%]+)")
BEGIN_ENV_RE = re.compile(r"\\begin\s*\{\s*([^\}]+)\s*\}")
END_ENV_RE = re.compile(r"\\end\s*\{\s*([^\}]+)\s*\}")

VERBATIM_ENVS = {
    "verbatim",
    "Verbatim",
    "lstlisting",
    "minted",
    "comment",
    "filecontents",
    "filecontents*",
}


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _ensure_file_exists(path: str, what: str) -> None:
    if not os.path.isfile(path):
        _die(f"{what} does not exist: {path}")


def _split_tex_comment(line: str) -> Tuple[str, str]:
    idx = line.find("%")
    while idx != -1:
        bs = 0
        j = idx - 1
        while j >= 0 and line[j] == "\\":
            bs += 1
            j -= 1
        if bs % 2 == 1:
            idx = line.find("%", idx + 1)
            continue
        return line[:idx], line[idx:]
    return line, ""


def strip_tex_comments(text: str) -> str:
    out_lines: List[str] = []
    for line in text.splitlines():
        code, _comment = _split_tex_comment(line)
        out_lines.append(code)
    return "\n".join(out_lines)


def _truncate_at_endinput(text: str) -> str:
    out: List[str] = []
    for line in text.splitlines(True):
        code, _comment = _split_tex_comment(line)
        if re.search(r"\\endinput\b", code):
            break
        out.append(line)
    return "".join(out)


def _extract_document_body_if_present(tex: str) -> str:
    m_begin = re.search(r"\\begin\s*\{\s*document\s*\}", tex, flags=re.IGNORECASE)
    if not m_begin:
        return tex
    m_end = re.search(r"\\end\s*\{\s*document\s*\}", tex, flags=re.IGNORECASE)
    if m_end and m_end.start() > m_begin.end():
        return tex[m_begin.end():m_end.start()]
    return tex[m_begin.end():]


def _normalize_input_target(target: str) -> str:
    t = target.strip().strip('"').strip("'")
    t = t.replace("\\", "/")
    while t.startswith("./"):
        t = t[2:]
    if t.startswith("/"):
        t = t[1:]
    return t


def _resolve_input_target(target: str, files_by_name: Dict[str, str]) -> Optional[str]:
    t = _normalize_input_target(target)
    candidates: List[str] = []
    candidates.append(t)
    if not t.lower().endswith(".tex"):
        candidates.append(t + ".tex")
    base = os.path.basename(t)
    candidates.append(base)
    if not base.lower().endswith(".tex"):
        candidates.append(base + ".tex")
    stem = os.path.splitext(base)[0]
    if stem:
        stem_matches = [fn for fn in files_by_name.keys() if os.path.splitext(fn)[0] == stem]
        if len(stem_matches) == 1:
            candidates.insert(0, stem_matches[0])

    for c in candidates:
        if c in files_by_name:
            return c
    return None


def _expand_inputs_in_tex(
    tex: str,
    files_by_name: Dict[str, str],
    include_stack: List[str],
    used_files: List[str],
    unresolved_inputs: Set[str],
    warnings: List[str],
) -> str:
    out: List[str] = []
    verb_stack: List[str] = []

    for raw_line in tex.splitlines(True):
        code, comment = _split_tex_comment(raw_line)

        if verb_stack:
            out.append(code + comment)
            m_end = END_ENV_RE.search(code)
            if m_end:
                env = m_end.group(1).strip()
                if env == verb_stack[-1]:
                    verb_stack.pop()
            continue

        m_begin = BEGIN_ENV_RE.search(code)
        if m_begin:
            env = m_begin.group(1).strip()
            if env in VERBATIM_ENVS:
                verb_stack.append(env)
                out.append(code + comment)
                continue

        m_endinput = re.search(r"\\endinput\b", code)
        if m_endinput:
            out.append(code[:m_endinput.start()] + comment)
            break

        pos = 0
        rebuilt: List[str] = []
        for m in INPUT_CMD_RE.finditer(code):
            rebuilt.append(code[pos:m.start()])

            cmd = m.group(1)
            raw_arg = m.group(2).strip()
            arg = raw_arg[1:-1] if raw_arg.startswith("{") and raw_arg.endswith("}") else raw_arg

            resolved = _resolve_input_target(arg, files_by_name)
            if resolved is None:
                unresolved_inputs.add(f"{cmd}:{arg}")
                rebuilt.append(code[m.start():m.end()])
            else:
                if resolved in include_stack:
                    warnings.append("cycle_detected:" + "->".join(include_stack + [resolved]))
                    rebuilt.append(code[m.start():m.end()])
                else:
                    expanded = _expand_file(resolved, files_by_name, include_stack, used_files, unresolved_inputs, warnings)
                    if cmd == "include":
                        rebuilt.append("\n\\clearpage\n" + expanded + "\n\\clearpage\n")
                    else:
                        rebuilt.append("\n" + expanded + "\n")

            pos = m.end()

        rebuilt.append(code[pos:])
        out.append("".join(rebuilt) + comment)

    return "".join(out)


def _expand_file(
    filename: str,
    files_by_name: Dict[str, str],
    include_stack: List[str],
    used_files: List[str],
    unresolved_inputs: Set[str],
    warnings: List[str],
) -> str:
    include_stack.append(filename)
    used_files.append(filename)

    tex = files_by_name.get(filename, "")
    tex = _truncate_at_endinput(tex)

    if len(include_stack) > 1:
        tex = _extract_document_body_if_present(tex)

    expanded = _expand_inputs_in_tex(tex, files_by_name, include_stack, used_files, unresolved_inputs, warnings)

    include_stack.pop()
    return expanded


def _candidate_root_score(tex: str) -> int:
    stripped = strip_tex_comments(tex)
    score = 0
    if DOCUMENTCLASS_RE.search(stripped):
        score += 1000
    if BEGIN_DOCUMENT_RE.search(stripped):
        score += 200
    if END_DOCUMENT_RE.search(stripped):
        score += 50
    score += min(len(tex) // 1000, 200)
    return score


def _merge_tex_bundle(
    filenames: List[str],
    latex_texts: List[str],
) -> Dict[str, object]:
    files_by_name: Dict[str, str] = {}
    for fn, tex in zip(filenames, latex_texts):
        if fn not in files_by_name:
            files_by_name[fn] = tex

    candidates: List[Tuple[int, str]] = []
    for fn, tex in files_by_name.items():
        stripped = strip_tex_comments(tex)
        if BEGIN_DOCUMENT_RE.search(stripped):
            candidates.append((_candidate_root_score(tex), fn))

    if not candidates:
        return {
            "ok": False,
            "error": "no_root_with_begin_document_found",
            "root": None,
            "merged": "",
            "used_files": [],
            "unresolved_inputs": [],
            "warnings": [],
        }

    candidates.sort(reverse=True)
    best_rank: Optional[Tuple[int, int, int, int]] = None
    best: Optional[Dict[str, object]] = None

    for root_score, root_fn in candidates:
        used_files: List[str] = []
        unresolved: Set[str] = set()
        warnings: List[str] = []

        merged = _expand_file(root_fn, files_by_name, [], used_files, unresolved, warnings)
        merged_stripped = strip_tex_comments(merged)
        has_theorem_in_merged = 1 if THEOREM_ENV_RE.search(merged_stripped) else 0

        used_unique: List[str] = list(dict.fromkeys(used_files))
        rank = (has_theorem_in_merged, len(used_unique), root_score, len(merged))

        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = {
                "ok": True,
                "error": None,
                "root": root_fn,
                "merged": merged,
                "used_files": used_unique,
                "unresolved_inputs": sorted(unresolved),
                "warnings": warnings,
            }

    assert best is not None
    return best


def _estimate_merged_perplexity(
    used_files: List[str],
    all_filenames: List[str],
    all_texts: List[str],
    all_ppls: List[float],
) -> Optional[float]:
    ppl_by_fn: Dict[str, float] = {}
    text_by_fn: Dict[str, str] = {}
    for fn, tx, ppl in zip(all_filenames, all_texts, all_ppls):
        if fn not in ppl_by_fn and isinstance(ppl, (int, float)):
            ppl_by_fn[fn] = float(ppl)
            text_by_fn[fn] = tx

    sum_w = 0.0
    sum_w_log = 0.0
    for fn in used_files:
        if fn not in ppl_by_fn:
            continue
        w = float(max(1, len(text_by_fn.get(fn, ""))))
        p = max(ppl_by_fn[fn], 1e-12)
        sum_w += w
        sum_w_log += w * math.log(p)

    if sum_w <= 0.0:
        return None
    return float(math.exp(sum_w_log / sum_w))


# =========================
# CHANGED: argparse interface
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metadata",  # CHANGED: was positional input_metadata_file
        dest="input_metadata_file",
        default=None,
        help="Optional JSONL metadata file; if provided, restricts to math.* categories and fills 'metadata'.",
    )
    p.add_argument(
        "--input",  # CHANGED: named argument
        dest="input_file",
        required=True,
        help="Input JSONL of TeX pieces.",
    )
    p.add_argument(
        "--output",  # CHANGED: named argument
        dest="output_file",
        required=True,
        help="Output JSONL of aggregated records.",
    )
    p.add_argument(
        "--merge",
        action="store_true",
        help="Attempt to merge multi-file TeX into a single TeX via \\input/\\include.",
    )
    return p.parse_args()


def _finalize_and_maybe_write(
    cur_d: Optional[dict],
    f_out,
    do_merge: bool,
    stats: dict,
) -> int:
    if cur_d is None:
        return 0

    if not cur_d.get("__has_begin_document", False):
        stats["skipped_no_begin_document"] += 1
        return 0
    if not cur_d.get("__has_theorem_env", False):
        stats["skipped_no_theorem_env"] += 1
        return 0

    if do_merge:
        merge_res = _merge_tex_bundle(cur_d["filenames"], cur_d["latex"])
        cur_d["merge"] = True
        cur_d["merge_root"] = merge_res.get("root")
        cur_d["merge_filenames"] = merge_res.get("used_files", [])
        cur_d["merge_unresolved_inputs"] = merge_res.get("unresolved_inputs", [])
        cur_d["merge_warnings"] = merge_res.get("warnings", [])
        if merge_res.get("ok", False):
            used_files = merge_res.get("used_files", [])
            merged_tex = merge_res.get("merged", "")

            cur_d["filenames"] = [merge_res.get("root") or "MERGED.tex"]
            cur_d["latex"] = [merged_tex]

            merged_ppl = _estimate_merged_perplexity(
                used_files=used_files,
                all_filenames=cur_d["filenames_source"],
                all_texts=cur_d["latex_source"],
                all_ppls=cur_d["kenlm_perplexity_source"],
            )
            merged_sp_ppl = _estimate_merged_perplexity(
                used_files=used_files,
                all_filenames=cur_d["filenames_source"],
                all_texts=cur_d["latex_source"],
                all_ppls=cur_d["kenlm_sp_perplexity_source"],
            )

            cur_d["kenlm_perplexity"] = [merged_ppl]
            cur_d["kenlm_sp_perplexity"] = [merged_sp_ppl]
        else:
            stats["merge_failed"] += 1

    cur_d.pop("__has_begin_document", None)
    cur_d.pop("__has_theorem_env", None)
    if not do_merge:
        cur_d.pop("filenames_source", None)
        cur_d.pop("latex_source", None)
        cur_d.pop("kenlm_perplexity_source", None)
        cur_d.pop("kenlm_sp_perplexity_source", None)

    print(json.dumps(cur_d), file=f_out)
    stats["written"] += 1
    return 1


def main() -> None:
    args = parse_args()

    # NEW: metadata is optional
    metadata: Optional[Dict[str, dict]] = None  # CHANGED: allow None

    if args.input_metadata_file is not None:  # NEW: only if provided
        _ensure_file_exists(args.input_metadata_file, "input_metadata_file")  # CHANGED: conditional
        metadata = {}
        with open(args.input_metadata_file, "r") as f:
            for raw in f:
                line = json.loads(raw)
                arxiv_id = line["id"]
                categories = line["categories"].split()
                has_math = any(cat.lower().startswith("math.") for cat in categories)
                if has_math:
                    metadata[arxiv_id] = line
        print(f"Loaded {len(metadata)} math papers' metadata")
    else:
        print("No metadata file provided; not restricting to math.* and 'metadata' will be None.")

    _ensure_file_exists(args.input_file, "input_file")

    stats = {
        "written": 0,
        "skipped_no_theorem_env": 0,
        "skipped_no_begin_document": 0,
        "merge_failed": 0,
    }

    cur_arxiv_id: Optional[str] = None
    cur_d: Optional[dict] = None

    with open(args.input_file, "r") as f_in, open(args.output_file, "w") as f_out:
        num_written = 0
        for i_line, raw in enumerate(f_in):
            if i_line % 10000 == 0:
                print(f"Processing line {i_line//10000}0K, {stats['written']} written")

            line = json.loads(raw)
            path = line["path"].split("/")
            if len(path) < 2:
                continue
            arxiv_id = path[-2]

            # CHANGED: only filter by metadata when metadata is available
            if metadata is not None and arxiv_id not in metadata:
                continue

            filename = path[-1]
            if not filename.endswith(".tex"):
                continue

            if (cur_arxiv_id is None) or (cur_arxiv_id != arxiv_id):
                if cur_d is not None:
                    num_written += _finalize_and_maybe_write(cur_d, f_out, args.merge, stats)
                cur_arxiv_id = arxiv_id
                cur_d = {
                    "arxiv_id": arxiv_id,
                    "metadata": metadata[arxiv_id] if metadata is not None else None,  # CHANGED: optional
                    "filenames": [],
                    "latex": [],
                    "kenlm_perplexity": [],
                    "kenlm_sp_perplexity": [],
                    "__has_theorem_env": False,
                    "__has_begin_document": False,
                    "filenames_source": [],
                    "latex_source": [],
                    "kenlm_perplexity_source": [],
                    "kenlm_sp_perplexity_source": [],
                }

            assert cur_d is not None

            tex_text = line.get("text", "")
            stripped = strip_tex_comments(tex_text)
            if BEGIN_DOCUMENT_RE.search(stripped):
                cur_d["__has_begin_document"] = True
            if THEOREM_ENV_RE.search(stripped):
                cur_d["__has_theorem_env"] = True

            cur_d["filenames_source"].append(filename)
            cur_d["latex_source"].append(tex_text)
            cur_d["kenlm_perplexity_source"].append(line.get("kenlm_perplexity"))
            cur_d["kenlm_sp_perplexity_source"].append(line.get("kenlm_sp_perplexity"))

            cur_d["filenames"].append(filename)
            cur_d["latex"].append(tex_text)
            cur_d["kenlm_perplexity"].append(line.get("kenlm_perplexity"))
            cur_d["kenlm_sp_perplexity"].append(line.get("kenlm_sp_perplexity"))

        num_written += _finalize_and_maybe_write(cur_d, f_out, args.merge, stats)

    print(
        "Done. "
        + f"written={stats['written']}, "
        + f"skipped_no_theorem_env={stats['skipped_no_theorem_env']}, "
        + f"skipped_no_begin_document={stats['skipped_no_begin_document']}, "
        + f"merge_failed={stats['merge_failed']}"
    )


if __name__ == "__main__":
    main()
