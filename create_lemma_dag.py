#!/usr/bin/env python3
r"""
Build theorem/lemma dependency DAGs from a JSONL of LaTeX sources.

INPUT JSONL (one record per line):
{
  "filenames": ["main.tex", "sec1.tex", ...],
  "latex":     ["<latex of main>", "<latex of sec1>", ...]
}

OUTPUT JSONL (one line per input line):
{
  "index": 0,
  "filenames": [...],
  "nodes": [
    {"id": "...", "env": "...", "label": "... or null", "file": "sec.tex", "ordinal": 1, "text": "<statement body>"},
    ...
  ],
  "edges": [{"from":"<target>", "to":"<dependency>"}, ...],  # U -> V means: U's proof uses V
  "forward_edges": [...],    # only if --emit-forward (forward refs shown separately)
  "adjacency": {"U":["V1","V2"]...},
  "topo_order": [...],
  "dag_valid": true/false,
  "unknown_refs": {"<label>": count, ...},
  "skipped_forward": <int>,
  "proofs": [ ... ]          # only if --debug-proofs
}

Key features
------------
- Two-pass parse: collect all theorem-like statements first (labels/text/positions), then attach proofs.
- Header-based target: \begin{proof}[Theorem~\ref{th:main}] reliably attaches to 'th:main'.
- Adds edges from both header refs (besides the target) and body refs.
- Robust ref matching:
    * direct (exact label or alias via --label-map)
    * canonical (lowercase + remove non-alphanumerics)
    * token-set fuzzy (Jaccard; enable via --fuzzy-labels)
- Options to include forward refs, emit forward refs separately, debug proofs, and include refs inside statements.

IMPORTANT CHANGE
----------------
For dependencies referenced inside a PROOF, the "earlier?" check now compares the
dependency's statement position to the PROOF'S position (not the target statement's).
This is crucial when the theorem is stated early but its proof appears later with lemmas in between.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from collections import deque

# ---------------------------
# Data model
# ---------------------------

@dataclass
class Node:
    ordinal: int
    id: str
    env: str
    file: str
    file_index: int
    start_offset: int
    label: Optional[str]
    text: str

# ---------------------------
# Light utilities
# ---------------------------

def remove_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", text)

def infer_envs_from_newtheorem(all_text: str, whitelist: Set[str]) -> List[str]:
    envs: Set[str] = set()
    pat = re.compile(r"\\newtheorem\*?\s*\{([A-Za-z@]+)\}\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", re.IGNORECASE)
    for m in pat.finditer(all_text):
        env_name = m.group(1).strip()
        display = re.sub(r"[^a-z]+", "", m.group(2).strip().lower())
        if display in whitelist:
            envs.add(env_name)
    return sorted(envs)

def build_ref_pattern(extra_cmds: Optional[List[str]] = None) -> re.Pattern:
    cmds = ["ref", "cref", "Cref", "autoref", "nameref", "namecref", "eqref"]
    if extra_cmds:
        cmds += [c for c in extra_cmds if c and isinstance(c, str)]
    part = "|".onejoin if False else "|".join  # keep lints calm
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
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out

def find_theorem_blocks(text: str, envs: List[str]) -> List[Tuple[int,int,str,str]]:
    blocks = []
    for env in envs:
        pat = re.compile(rf"\\begin\{{{re.escape(env)}\*?\}}\s*.*?\s*\\end\{{{re.escape(env)}\*?\}}", re.DOTALL)
        for m in pat.finditer(text):
            blocks.append((m.start(), m.end(), env, m.group(0)))
    blocks.sort(key=lambda t: t[0])
    return blocks

def find_proof_blocks(text: str) -> List[Tuple[int,int,Optional[str],str,str]]:
    pat = re.compile(r"\\begin\{proof\*?\}(?:\[(?P<header>[^\]]*)\])?(?P<body>.*?)\\end\{proof\*?\}", re.DOTALL)
    out = []
    for m in pat.finditer(text):
        out.append((m.start(), m.end(), m.group("header"), m.group("body"), m.group(0)))
    return out

def extract_label(block: str) -> Optional[str]:
    m = re.search(r"\\label\{([^}]+)\}", block)
    return m.group(1).strip() if m else None

def extract_env_inner_text(block: str, env: str) -> str:
    pat = re.compile(rf"^\s*\\begin\{{{re.escape(env)}\*?\}}\s*(?:\[[^\]]*\]\s*)?(?P<body>.*)\s*\\end\{{{re.escape(env)}\*?\}}\s*$", re.DOTALL)
    m = pat.match(block)
    if not m:
        body = block
        body = re.sub(rf"^\s*\\begin\{{{re.escape(env)}\*?\}}\s*", "", body, flags=re.DOTALL)
        body = re.sub(rf"\\end\{{{re.escape(env)}\*?\}}\s*$", "", body, flags=re.DOTALL)
    else:
        body = m.group("body")
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    return body.strip()

def scan_events(cleaned_texts: List[Tuple[str, str]], envs: List[str]) -> List[Dict]:
    events = []
    for fi, (name, text) in enumerate(cleaned_texts):
        for start, end, env, block in find_theorem_blocks(text, envs):
            events.append({"type":"thm","file_index":fi,"start":start,"env":env,"block":block,"file":name})
        for start, end, hdr, body, block in find_proof_blocks(text):
            events.append({"type":"prf","file_index":fi,"start":start,"header":hdr,"body":body,"block":block,"file":name})
    events.sort(key=lambda e: (e["file_index"], e["start"]))
    return events

def normalize_label_case(label: Optional[str], lowercase: bool) -> Optional[str]:
    if label is None:
        return None
    return label.casefold() if lowercase else label

def canonicalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())

def label_tokens(label: str) -> Set[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label)
    parts = re.split(r"[:_/.\- ]+", s)
    tokens = {p.strip().lower() for p in parts if p.strip()}
    return tokens

def build_label_indices(nodes: List[Node], lowercase_labels: bool, label_map: Optional[Dict[str,str]]) -> Tuple[Dict[str,int], Dict[str,Set[int]], Dict[int,Set[str]]]:
    direct: Dict[str,int] = {}
    canonical_map: Dict[str, Set[int]] = {}
    tokens_map: Dict[int, Set[str]] = {}
    for idx, n in enumerate(nodes):
        if n.label:
            norm = normalize_label_case(n.label, lowercase_labels)
            if norm is not None and norm not in direct:
                direct[norm] = idx
            if norm is not None:
                canonical_map.setdefault(canonicalize_label(norm), set()).add(idx)
                tokens_map[idx] = label_tokens(norm)
    if label_map:
        for alias, target in label_map.items():
            alias_norm = normalize_label_case(alias, lowercase_labels)
            target_norm = normalize_label_case(target, lowercase_labels)
            if target_norm in direct and alias_norm not in direct:
                direct[alias_norm] = direct[target_norm]
                canonical_map.setdefault(canonicalize_label(alias_norm), set()).add(direct[target_norm])
    return direct, canonical_map, tokens_map

def resolve_ref(raw_ref: str, direct_map: Dict[str,int], canonical_map: Dict[str, Set[int]], tokens_map: Dict[int, Set[str]], lowercase_labels: bool, label_map: Optional[Dict[str,str]], fuzzy: bool, fuzzy_threshold: float) -> Tuple[Optional[int], str]:
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
    ref_tokens = label_tokens(norm)
    best_idx = None
    best_score = 0.0
    for idx, toks in tokens_map.items():
        if not toks:
            continue
        inter = len(ref_tokens & toks)
        union = len(ref_tokens | toks)
        if union == 0:
            continue
        score = inter / union
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is not None and best_score >= fuzzy_threshold:
        return best_idx, "fuzzy"
    return None, "unknown"

def is_earlier(a_file_index: int, a_start: int, b_file_index: int, b_start: int) -> bool:
    return (a_file_index, a_start) < (b_file_index, b_start)

def dedupe_edges(edges: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    seen = set()
    out = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

def topological_sort_ids(ids: List[str], edges: List[Tuple[str,str]]) -> Tuple[List[str], bool, Dict[str, List[str]]]:
    indeg = {i:0 for i in ids}
    adj = {i:[] for i in ids}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        indeg[v] = indeg.get(v, 0) + 1
    q = deque([i for i in ids if indeg.get(i,0)==0])
    topo = []
    count = 0
    while q:
        u = q.popleft()
        topo.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
        count += 1
    dag_valid = (count == len(ids))
    return topo, dag_valid, adj

def pass1_collect_nodes(events: List[Dict]) -> Tuple[List[Node], Dict[Tuple[int,int], int]]:
    nodes = []
    key_to_idx = {}
    ordinal = 0
    for ev in events:
        if ev["type"] != "thm":
            continue
        env = ev["env"]
        block = ev["block"]
        label = extract_label(block)
        text = extract_env_inner_text(block, env)
        ordinal += 1
        basename = os.path.basename(ev["file"])
        auto_id = f"{env}#{ordinal}@{basename}"
        node_id = label if label else auto_id
        node = Node(ordinal=ordinal, id=node_id, env=env, file=ev["file"], file_index=ev["file_index"], start_offset=ev["start"], label=label, text=text)
        nodes.append(node)
        idx = len(nodes)-1
        key_to_idx[(ev["file_index"], ev["start"])] = idx
    return nodes, key_to_idx

def build_edges(
    events: List[Dict],
    nodes: List[Node],
    key_to_idx: Dict[Tuple[int,int], int],
    lowercase_labels: bool,
    label_map: Optional[Dict[str,str]],
    include_forward: bool,
    emit_forward: bool,
    refpat: re.Pattern,
    fuzzy: bool,
    fuzzy_threshold: float,
    debug_proofs: bool
) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]], Dict[str,int], int, List[Dict]]:
    edges: List[Tuple[str,str]] = []
    forward_edges: List[Tuple[str,str]] = []
    unknown_refs: Dict[str,int] = {}
    skipped_forward = 0
    proof_assigned = {i: False for i in range(len(nodes))}
    pending_stack: List[int] = []
    direct_map, canonical_map, tokens_map = build_label_indices(nodes, lowercase_labels, label_map)
    proof_diags: List[Dict] = []

    for ev in events:
        if ev["type"] == "thm":
            idx = key_to_idx[(ev["file_index"], ev["start"])]
            pending_stack.append(idx)
            continue

        # PROOF event
        hdr = ev.get("header") or ""
        body = ev.get("body") or ""
        hdr_refs_raw = extract_refs_from_text(hdr, refpat)

        # Resolve target from header refs
        target_idx = None
        target_label_norm = None
        method_used = None
        for raw in hdr_refs_raw:
            idx_cand, method = resolve_ref(raw, direct_map, canonical_map, tokens_map, lowercase_labels, label_map, fuzzy, fuzzy_threshold)
            if idx_cand is not None:
                target_idx = idx_cand
                target_label_norm = normalize_label_case(nodes[idx_cand].label, lowercase_labels) if nodes[idx_cand].label else None
                method_used = method
                break

        # Fallback: most recent theorem without a proof
        if target_idx is None:
            while pending_stack:
                cand = pending_stack.pop()
                if not proof_assigned[cand]:
                    target_idx = cand
                    method_used = "fallback"
                    break

        diag = None
        if debug_proofs:
            diag = {
                "file": ev["file"],
                "file_index": ev["file_index"],
                "offset": ev["start"],
                "header": hdr,
                "header_refs_raw": hdr_refs_raw,
                "target_idx": target_idx,
                "target_id": (nodes[target_idx].id if target_idx is not None else None),
                "target_method": method_used,
                "edges_emitted": [],
                "forward_edges": [],
                "unknown_header": [],
                "unknown_body": [],
                "body_excerpt": (body[:200] + ("..." if len(body)>200 else "")),
            }

        if target_idx is None:
            for raw in hdr_refs_raw:
                idx_cand, _ = resolve_ref(raw, direct_map, canonical_map, tokens_map, lowercase_labels, label_map, fuzzy, fuzzy_threshold)
                if idx_cand is None:
                    unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
                    if debug_proofs and diag:
                        diag["unknown_header"].append(raw)
            if debug_proofs and diag:
                proof_diags.append(diag)
            continue

        proof_assigned[target_idx] = True
        if target_idx in pending_stack:
            pending_stack.remove(target_idx)

        target_node = nodes[target_idx]

        # Anchor "earlier?" to the PROOF position (not target statement)
        proof_fi = ev["file_index"]          # ### CHANGED (proof-anchored earlier check)
        proof_start = ev["start"]            # ### CHANGED (proof-anchored earlier check)

        # Header-derived dependencies (besides the target)
        for raw in hdr_refs_raw:
            idx_dep, method_dep = resolve_ref(raw, direct_map, canonical_map, tokens_map, lowercase_labels, label_map, fuzzy, fuzzy_threshold)
            if idx_dep is None:
                unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
                if debug_proofs and diag:
                    diag["unknown_header"].append(raw)
                continue
            if idx_dep == target_idx:
                continue
            dep_node = nodes[idx_dep]
            # earlier = w.r.t. PROOF occurrence, not target statement
            earlier = is_earlier(dep_node.file_index, dep_node.start_offset, proof_fi, proof_start)  # ### CHANGED (proof-anchored earlier check)
            edge = (target_node.id, dep_node.id)
            if earlier or include_forward:
                edges.append(edge)
                if debug_proofs and diag:
                    diag["edges_emitted"].append({"from": edge[0], "to": edge[1], "kind": "header", "forward": (not earlier), "method": method_dep})
                if (not earlier) and emit_forward:
                    forward_edges.append(edge)
                    if debug_proofs and diag:
                        diag["forward_edges"].append({"from": edge[0], "to": edge[1], "kind": "header"})
            else:
                skipped_forward += 1
                if emit_forward:
                    forward_edges.append(edge)
                if debug_proofs and diag:
                    diag["forward_edges"].append({"from": edge[0], "to": edge[1], "kind": "header"})

        # Body-derived dependencies
        body_refs_raw = extract_refs_from_text(body, refpat)
        if debug_proofs and diag:
            diag["body_refs_raw"] = body_refs_raw
        seen_dep_ids: Set[str] = set()
        for raw in body_refs_raw:
            idx_dep, method_dep = resolve_ref(raw, direct_map, canonical_map, tokens_map, lowercase_labels, label_map, fuzzy, fuzzy_threshold)
            if idx_dep is None:
                unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
                if debug_proofs and diag:
                    diag["unknown_body"].append(raw)
                continue
            if nodes[idx_dep].id in seen_dep_ids:
                continue
            seen_dep_ids.add(nodes[idx_dep].id)
            if target_node.label and normalize_label_case(target_node.label, lowercase_labels) == normalize_label_case(nodes[idx_dep].label, lowercase_labels):
                continue  # ignore self-ref
            dep_node = nodes[idx_dep]
            # earlier = w.r.t. PROOF occurrence, not target statement
            earlier = is_earlier(dep_node.file_index, dep_node.start_offset, proof_fi, proof_start)  # ### CHANGED (proof-anchored earlier check)
            edge = (target_node.id, dep_node.id)
            if earlier or include_forward:
                edges.append(edge)
                if debug_proofs and diag:
                    diag["edges_emitted"].append({"from": edge[0], "to": edge[1], "kind": "body", "forward": (not earlier), "method": method_dep})
                if (not earlier) and emit_forward:
                    forward_edges.append(edge)
                    if debug_proofs and diag:
                        diag["forward_edges"].append({"from": edge[0], "to": edge[1], "kind": "body"})
            else:
                skipped_forward += 1
                if emit_forward:
                    forward_edges.append(edge)
                if debug_proofs and diag:
                    diag["forward_edges"].append({"from": edge[0], "to": edge[1], "kind": "body"})

        if debug_proofs and diag:
            proof_diags.append(diag)

    edges = dedupe_edges(edges)
    forward_edges = dedupe_edges(forward_edges)
    return edges, forward_edges, unknown_refs, skipped_forward, proof_diags

def build_adjacency(edges: List[Tuple[str,str]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    return adj

def process_record(
    filenames: List[str],
    latex_list: List[str],
    envs: List[str],
    include_forward: bool,
    emit_forward: bool,
    lowercase_labels: bool,
    label_map: Optional[Dict[str,str]],
    ref_commands: List[str],
    fuzzy: bool,
    fuzzy_threshold: float,
    include_statement_refs: bool,
    debug_proofs: bool
) -> Dict:
    if len(filenames) != len(latex_list):
        raise ValueError("filenames and latex arrays must have equal length")
    cleaned_texts = []
    for i in range(len(filenames)):
        name = filenames[i]
        text = latex_list[i] if latex_list[i] is not None else ""
        cleaned_texts.append((name, remove_comments(text)))
    whitelist = {"lemma", "theorem", "proposition", "corollary", "claim"}
    all_text = "\n".join(t for _, t in cleaned_texts)
    inferred = infer_envs_from_newtheorem(all_text, whitelist)
    envs_merged = envs + [e for e in inferred if e not in envs]
    events = scan_events(cleaned_texts, envs_merged)
    nodes, key_to_idx = pass1_collect_nodes(events)
    refpat = build_ref_pattern(ref_commands)
    edges, forward_edges, unknown_refs, skipped_forward, proof_diags = build_edges(
        events, nodes, key_to_idx, lowercase_labels, label_map, include_forward, emit_forward, refpat, fuzzy, fuzzy_threshold, debug_proofs
    )
    # Optionally extract dependencies from inside statements (anchored to statement position)
    if include_statement_refs and nodes:
        direct_map, canonical_map, tokens_map = build_label_indices(nodes, lowercase_labels, label_map)
        for n in nodes:
            srefs = extract_refs_from_text(n.text, refpat)
            for raw in srefs:
                idx_dep, _ = resolve_ref(raw, direct_map, canonical_map, tokens_map, lowercase_labels, label_map, fuzzy, fuzzy_threshold)
                if idx_dep is None:
                    unknown_refs[raw] = unknown_refs.get(raw, 0) + 1
                    continue
                dep_node = nodes[idx_dep]
                if n.label and normalize_label_case(n.label, lowercase_labels) == normalize_label_case(dep_node.label, lowercase_labels):
                    continue
                earlier = is_earlier(dep_node.file_index, dep_node.start_offset, n.file_index, n.start_offset)
                edge = (n.id, dep_node.id)
                if earlier or include_forward:
                    edges.append(edge)
                else:
                    skipped_forward += 1
                    if emit_forward:
                        forward_edges.append(edge)
        edges = dedupe_edges(edges)
        if emit_forward:
            forward_edges = dedupe_edges(forward_edges)

    ids = [n.id for n in nodes]
    topo, dag_valid, adjacency = topological_sort_ids(ids, edges)
    out = {
        "filenames": filenames,
        "nodes": [{"id": n.id, "env": n.env, "label": n.label, "file": n.file, "ordinal": n.ordinal, "text": n.text} for n in nodes],
        "edges": [{"from": u, "to": v} for (u, v) in edges],
        "adjacency": adjacency,
        "topo_order": topo,
        "dag_valid": dag_valid,
        "unknown_refs": unknown_refs,
        "skipped_forward": skipped_forward
    }
    if emit_forward:
        out["forward_edges"] = [{"from": u, "to": v} for (u, v) in forward_edges]
    if debug_proofs:
        out["proofs"] = proof_diags
    return out

def process_jsonl(
    input_path: str,
    output_path: str,
    envs: List[str],
    include_forward: bool,
    emit_forward: bool,
    lowercase_labels: bool,
    label_map_path: Optional[str],
    ref_commands: List[str],
    fuzzy: bool,
    fuzzy_threshold: float,
    include_statement_refs: bool,
    debug_proofs: bool
) -> Tuple[int,int]:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    label_map = None
    if label_map_path:
        if not os.path.isfile(label_map_path):
            raise FileNotFoundError(f"Label map not found: {label_map_path}")
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
            if not isinstance(label_map, dict):
                raise ValueError("--label-map must be a JSON object")
    records = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                records.append(line.rstrip("\n"))
    outputs = []
    errors = 0
    for idx, raw in enumerate(records):
        try:
            rec = json.loads(raw)
        except Exception as e:
            outputs.append(json.dumps({"index": idx, "error": f"JSON parse error: {e}"}))
            errors += 1
            continue
        filenames = rec.get("filenames")
        latex = rec.get("latex")
        if not isinstance(filenames, list) or not isinstance(latex, list):
            outputs.append(json.dumps({"index": idx, "error": "Record must contain 'filenames' (list) and 'latex' (list)."}))
            errors += 1
            continue
        try:
            out = process_record(filenames, latex, envs, include_forward, emit_forward, lowercase_labels, label_map, ref_commands, fuzzy, fuzzy_threshold, include_statement_refs, debug_proofs)
            out["index"] = idx
            outputs.append(json.dumps(out, ensure_ascii=False))
        except Exception as e:
            outputs.append(json.dumps({"index": idx, "filenames": filenames, "error": f"Processing error: {e}"}))
            errors += 1
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for line in outputs:
            fout.write(line + "\n")
    os.replace(tmp_path, output_path)
    return len(records), errors

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build theorem/lemma dependency DAGs from JSONL of LaTeX sources.")
    p.add_argument("-i","--input",required=True,help="Path to input JSONL")
    p.add_argument("-o","--output",required=True,help="Path to output JSONL")
    p.add_argument("--envs",default="lemma,theorem,proposition,corollary,claim",help="Comma-separated theorem-like environments to scan")
    p.add_argument("--include-forward",action="store_true",help="Include edges to forward references (may create cycles)")
    p.add_argument("--emit-forward",action="store_true",help="Also emit forward edges under 'forward_edges'")
    p.add_argument("--debug-proofs",action="store_true",help="Emit per-proof diagnostics under 'proofs'")
    p.add_argument("--label-map",help="Path to JSON file mapping alias->canonical labels to remap refs")
    p.add_argument("--lowercase-labels",action="store_true",help="Match labels case-insensitively by lowercasing")
    p.add_argument("--refcmds",default="ref,cref,Cref,autoref,nameref,namecref,eqref",help="Comma-separated ref-like commands to recognize")
    p.add_argument("--fuzzy-labels",action="store_true",help="Enable fuzzy matching for ref labels (canonical + token Jaccard)")
    p.add_argument("--fuzzy-threshold",type=float,default=0.66,help="Threshold (0-1) for token-set fuzzy matching")
    p.add_argument("--include-statement-refs",action="store_true",help="Also treat refs inside theorem/lemma statements as dependencies")
    return p.parse_args(argv if argv is not None else sys.argv[1:])

def main():
    args = parse_args()
    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    refcmds = [c.strip() for c in args.refcmds.split(",") if c.strip()]
    try:
        nrec, nerr = process_jsonl(
            input_path=args.input,
            output_path=args.output,
            envs=envs,
            include_forward=args.include_forward,
            emit_forward=args.emit_forward,
            lowercase_labels=args.lowercase_labels,
            label_map_path=args.label_map,
            ref_commands=refcmds,
            fuzzy=args.fuzzy_labels,
            fuzzy_threshold=args.fuzzy_threshold,
            include_statement_refs=args.include_statement_refs,
            debug_proofs=args.debug_proofs
        )
        print(f"[ok] Processed {nrec} record(s). Wrote: {args.output}")
        if nerr:
            print(f"[warn] {nerr} record(s) had errors; see output JSONL for details.")
        sys.exit(0)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
