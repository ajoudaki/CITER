
#!/usr/bin/env python3
r"""
Read a DAG JSONL (output of dag_creator.py) and, for each record, print a JSON list
of (source_id, source_text, dest_id, dest_text) for every edge.

Streaming:
- Reads input JSONL line-by-line.
- Does NOT store all records.
- Does NOT store all edges; prints each record’s edge list incrementally.
- Builds an id->text map per record (needed to attach texts).

Output (one line per input record):
[
  ["src_id", "src_text", "dst_id", "dst_text"],
  ...
]

If an input record contains an "error" field, this script prints [] for that record.
"""

import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple, TextIO


EdgeQuad = Tuple[str, Optional[str], str, Optional[str]]


class EdgeTextIterator:
    """
    Iterator over edges that yields (src_id, src_text, dst_id, dst_text).

    NOTE: src_id/dst_id refer to the DAG output's edge direction.
    In your current dag_creator.py, edge u -> v means: v depends on u.
    """

    def __init__(self, edges: List[dict], id_to_text: Dict[str, str]) -> None:
        self._edges = edges
        self._id_to_text = id_to_text
        self._i = 0

    def __iter__(self) -> "EdgeTextIterator":
        return self

    def __next__(self) -> EdgeQuad:
        while self._i < len(self._edges):
            e = self._edges[self._i]
            self._i += 1

            src = e.get("from")
            dst = e.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue

            return (src, self._id_to_text.get(src), dst, self._id_to_text.get(dst))

        raise StopIteration


class RecordEdgeStream:
    """
    Streaming reader over a dag_creator output JSONL.

    Yields:
      (record_index, EdgeTextIterator)
    """

    def __init__(self, infile: TextIO) -> None:
        self._infile = infile

    def __iter__(self) -> Iterator[Tuple[Optional[int], EdgeTextIterator]]:
        for raw_line in self._infile:
            line = raw_line.strip()
            if not line:
                continue

            obj = self._safe_json_load(line)
            if obj is None or not isinstance(obj, dict):
                yield (None, EdgeTextIterator([], {}))
                continue

            idx = obj.get("index") if isinstance(obj.get("index"), int) else None

            if "error" in obj:
                yield (idx, EdgeTextIterator([], {}))
                continue

            nodes = obj.get("nodes", [])
            edges = obj.get("edges", [])

            id_to_text = self._build_id_to_text(nodes)
            edge_dicts = self._normalize_edges(edges)

            yield (idx, EdgeTextIterator(edge_dicts, id_to_text))

    @staticmethod
    def _safe_json_load(s: str) -> Optional[dict]:
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def _build_id_to_text(nodes: object) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not isinstance(nodes, list):
            return out
        for n in nodes:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            txt = n.get("text")
            if isinstance(nid, str) and isinstance(txt, str):
                out[nid] = txt
        return out

    @staticmethod
    def _normalize_edges(edges: object) -> List[dict]:
        if not isinstance(edges, list):
            return []
        out: List[dict] = []
        for e in edges:
            if isinstance(e, dict):
                out.append(e)
        return out


def ensure_readable_file(path: str, desc: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")


def write_edge_list_as_json_array(
    out: TextIO,
    edge_iter: Iterator[EdgeQuad],
    include_index: bool,
    index_value: Optional[int],
) -> None:
    """
    Prints one JSON value per record.

    If include_index=False (default): prints a JSON array of 4-tuples.
    If include_index=True: prints {"index": <idx|null>, "edges": [ ... ]}.
    """
    if include_index:
        out.write('{"index":')
        out.write("null" if index_value is None else str(index_value))
        out.write(',"edges":')
    out.write("[")

    first = True
    for src_id, src_text, dst_id, dst_text in edge_iter:
        item = [src_id, src_text, dst_id, dst_text]
        if first:
            first = False
            out.write(json.dumps(item, ensure_ascii=False))
        else:
            out.write("," + json.dumps(item, ensure_ascii=False))

    out.write("]")
    if include_index:
        out.write("}")
    out.write("\n")
    out.flush()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream DAG JSONL and print per-record edge quadruples (id/text pairs)."
    )
    p.add_argument(
        "-i", "--input", required=True, help="Path to dag_creator output JSONL"
    )
    p.add_argument(
        "-o", "--output", default="-", help="Output path (default stdout). Use '-' for stdout."
    )
    p.add_argument(
        "--include-index",
        action="store_true",
        help='Wrap each line as {"index": ..., "edges": [...]} instead of just printing the edges list.',
    )
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    ensure_readable_file(args.input, "Input JSONL")

    if args.output == "-":
        fout: TextIO = sys.stdout
    else:
        fout = open(args.output, "w", encoding="utf-8")

    try:
        with open(args.input, "r", encoding="utf-8") as fin:
            stream = RecordEdgeStream(fin)
            for idx, edge_iter in stream:
                write_edge_list_as_json_array(
                    out=fout,
                    edge_iter=edge_iter,
                    include_index=bool(args.include_index),
                    index_value=idx,
                )
    finally:
        if args.output != "-":
            fout.close()


if __name__ == "__main__":
    main()
