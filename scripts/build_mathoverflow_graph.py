"""
Build a Q&A graph from StackExchange data dumps.

Supports multiple sites: math, mathoverflow, stats, etc.

Edges:
  - Question -> Accepted Answer
  - Question -> High-voted Answer (score >= threshold)
  - Post -> Linked Post (when one post references another)
  - Question -> Duplicate Question

Usage:
  python build_mathoverflow_graph.py --archives math mathoverflow stats [--sample N] [--output PATH]
"""

import py7zr
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Iterator
from collections import defaultdict
import json
import argparse
from pathlib import Path


@dataclass
class Post:
    id: str  # Now string with site prefix: "math_123"
    post_type: int  # 1=Question, 2=Answer
    parent_id: Optional[str] = None  # For answers: question ID
    accepted_answer_id: Optional[str] = None  # For questions: accepted answer
    score: int = 0
    title: str = ""
    body: str = ""
    source: str = ""  # Site name
    url: str = ""  # Direct link to post
    tags: list = field(default_factory=list)  # For questions only
    creation_date: str = ""
    last_activity_date: str = ""
    last_edit_date: str = ""
    view_count: int = 0
    answer_count: int = 0
    comment_count: int = 0


def parse_tags(tags_str: str) -> list[str]:
    """Parse tags from format '<tag1><tag2><tag3>' to ['tag1', 'tag2', 'tag3']."""
    if not tags_str:
        return []
    # Remove leading/trailing < > and split
    return [t for t in tags_str.replace('>', '').split('<') if t]


@dataclass
class PostLink:
    post_id: str
    related_post_id: str
    link_type: int  # 1=Linked, 3=Duplicate


@dataclass
class Graph:
    """Simple directed graph with labeled edges."""
    edges: dict = field(default_factory=lambda: defaultdict(list))
    nodes: dict = field(default_factory=dict)

    def add_edge(self, src: str, dst: str, label: str):
        self.edges[src].append((dst, label))

    def add_node(self, node_id: str, data: dict):
        self.nodes[node_id] = data

    def stats(self) -> dict:
        edge_counts = defaultdict(int)
        for src, dsts in self.edges.items():
            for dst, label in dsts:
                edge_counts[label] += 1

        # Count nodes by source
        source_counts = defaultdict(int)
        for node_id, data in self.nodes.items():
            source_counts[data.get('source', 'unknown')] += 1

        return {
            "num_nodes": len(self.nodes),
            "num_edges": sum(edge_counts.values()),
            "edges_by_type": dict(edge_counts),
            "nodes_by_source": dict(source_counts)
        }

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": {str(k): v for k, v in self.edges.items()},
            "stats": self.stats()
        }


def iter_xml_rows(xml_path: str, max_rows: Optional[int] = None) -> Iterator[dict]:
    """Iterate over <row> elements in XML file."""
    count = 0
    for event, elem in ET.iterparse(xml_path, events=['end']):
        if elem.tag == 'row':
            yield elem.attrib
            elem.clear()
            count += 1
            if max_rows and count >= max_rows:
                break


def parse_posts(xml_path: str, site_prefix: str, base_url: str, max_rows: Optional[int] = None) -> tuple[dict, dict]:
    """Parse Posts.xml, return (questions, answers) dicts."""
    questions = {}  # id -> Post
    answers = {}    # id -> Post

    for row in iter_xml_rows(xml_path, max_rows):
        post_type = int(row.get('PostTypeId', 0))
        if post_type not in (1, 2):
            continue

        raw_id = row['Id']
        post_id = f"{site_prefix}_{raw_id}"
        parent_id = f"{site_prefix}_{row['ParentId']}" if 'ParentId' in row else None
        accepted_id = f"{site_prefix}_{row['AcceptedAnswerId']}" if 'AcceptedAnswerId' in row else None

        # Construct URL: questions use /questions/ID, answers use /a/ID
        if post_type == 1:
            url = f"{base_url}/questions/{raw_id}"
        else:
            url = f"{base_url}/a/{raw_id}"

        post = Post(
            id=post_id,
            post_type=post_type,
            parent_id=parent_id,
            accepted_answer_id=accepted_id,
            score=int(row.get('Score', 0)),
            title=row.get('Title', ''),
            body=row.get('Body', ''),
            source=site_prefix,
            url=url,
            tags=parse_tags(row.get('Tags', '')) if post_type == 1 else [],
            creation_date=row.get('CreationDate', ''),
            last_activity_date=row.get('LastActivityDate', ''),
            last_edit_date=row.get('LastEditDate', ''),
            view_count=int(row.get('ViewCount', 0)),
            answer_count=int(row.get('AnswerCount', 0)),
            comment_count=int(row.get('CommentCount', 0))
        )

        if post_type == 1:
            questions[post.id] = post
        else:
            answers[post.id] = post

    return questions, answers


def parse_post_links(xml_path: str, site_prefix: str, max_rows: Optional[int] = None) -> list[PostLink]:
    """Parse PostLinks.xml for linked and duplicate links."""
    links = []
    for row in iter_xml_rows(xml_path, max_rows):
        link_type = int(row.get('LinkTypeId', 0))
        if link_type in (1, 3):  # 1=Linked, 3=Duplicate
            links.append(PostLink(
                post_id=f"{site_prefix}_{row['PostId']}",
                related_post_id=f"{site_prefix}_{row['RelatedPostId']}",
                link_type=link_type
            ))
    return links


def build_graph(questions: dict, answers: dict, post_links: list,
                vote_threshold: int = 5) -> Graph:
    """Build the Q&A graph."""
    graph = Graph()

    # Add question nodes
    for qid, q in questions.items():
        graph.add_node(qid, {
            "type": "question",
            "title": q.title,
            "body": q.body,
            "score": q.score,
            "source": q.source,
            "url": q.url,
            "tags": q.tags,
            "creation_date": q.creation_date,
            "last_activity_date": q.last_activity_date,
            "last_edit_date": q.last_edit_date,
            "view_count": q.view_count,
            "answer_count": q.answer_count,
            "comment_count": q.comment_count
        })

    # Add answer nodes and edges
    for aid, a in answers.items():
        if a.parent_id not in questions:
            continue

        graph.add_node(aid, {
            "type": "answer",
            "body": a.body,
            "score": a.score,
            "parent_id": a.parent_id,
            "source": a.source,
            "url": a.url,
            "creation_date": a.creation_date,
            "last_activity_date": a.last_activity_date,
            "last_edit_date": a.last_edit_date,
            "comment_count": a.comment_count
        })

        q = questions[a.parent_id]

        # Edge: Question -> Accepted Answer
        if q.accepted_answer_id == aid:
            graph.add_edge(q.id, aid, "accepted_answer")

        # Edge: Question -> High-voted Answer
        if a.score >= vote_threshold:
            graph.add_edge(q.id, aid, "voted_answer")

    # Edge: Post -> Post (Linked or Duplicate)
    for link in post_links:
        # Only add if both posts exist in our graph
        if link.post_id in graph.nodes and link.related_post_id in graph.nodes:
            if link.link_type == 1:
                graph.add_edge(link.post_id, link.related_post_id, "linked")
            elif link.link_type == 3:
                graph.add_edge(link.post_id, link.related_post_id, "duplicate")

    return graph


# Map short names to archive files and base URLs
SITE_INFO = {
    "math": {
        "archive": "math.stackexchange.com.7z",
        "url": "https://math.stackexchange.com"
    },
    "mathoverflow": {
        "archive": "mathoverflow.net.7z",
        "url": "https://mathoverflow.net"
    },
    "stats": {
        "archive": "stats.stackexchange.com.7z",
        "url": "https://stats.stackexchange.com"
    },
    "tex": {
        "archive": "tex.stackexchange.com.7z",
        "url": "https://tex.stackexchange.com"
    },
    "askubuntu": {
        "archive": "askubuntu.com.7z",
        "url": "https://askubuntu.com"
    },
    "superuser": {
        "archive": "superuser.com.7z",
        "url": "https://superuser.com"
    },
    "law": {
        "archive": "law.stackexchange.com.7z",
        "url": "https://law.stackexchange.com"
    },
}

# For backward compatibility
SITE_ARCHIVES = {k: v["archive"] for k, v in SITE_INFO.items()}


def main():
    parser = argparse.ArgumentParser(description="Build StackExchange Q&A graph")
    parser.add_argument("--archives", nargs="+", default=["mathoverflow"],
                        choices=list(SITE_ARCHIVES.keys()),
                        help="Sites to process (default: mathoverflow)")
    parser.add_argument("--archive-dir", default="data/SE",
                        help="Directory containing 7z archives")
    parser.add_argument("--sample", type=int, default=None,
                        help="Max rows to parse per site (for testing)")
    parser.add_argument("--vote-threshold", type=int, default=2,
                        help="Min score for voted_answer edges")
    parser.add_argument("--output", default="data/SE/se_graph.json",
                        help="Output JSON path")
    args = parser.parse_args()

    all_questions = {}
    all_answers = {}
    all_links = []

    for site in args.archives:
        site_info = SITE_INFO[site]
        archive_path = Path(args.archive_dir) / site_info["archive"]
        base_url = site_info["url"]
        extract_dir = f"/tmp/se_extract_{site}"

        print(f"\n{'='*60}")
        print(f"Processing: {site}")
        print(f"{'='*60}")

        print(f"Loading archive: {archive_path}")
        with py7zr.SevenZipFile(archive_path, 'r') as z:
            z.extractall(path=extract_dir)

        print(f"Parsing Posts.xml (sample={args.sample})...")
        questions, answers = parse_posts(f'{extract_dir}/Posts.xml', site, base_url, args.sample)
        print(f"  Questions: {len(questions):,}, Answers: {len(answers):,}")

        print(f"Parsing PostLinks.xml...")
        post_links = parse_post_links(f'{extract_dir}/PostLinks.xml', site, args.sample)
        linked_count = sum(1 for l in post_links if l.link_type == 1)
        duplicate_count = sum(1 for l in post_links if l.link_type == 3)
        print(f"  Linked: {linked_count:,}, Duplicate: {duplicate_count:,}")

        all_questions.update(questions)
        all_answers.update(answers)
        all_links.extend(post_links)

    print(f"\n{'='*60}")
    print(f"Building combined graph (vote_threshold={args.vote_threshold})...")
    print(f"{'='*60}")
    print(f"Total questions: {len(all_questions):,}")
    print(f"Total answers: {len(all_answers):,}")
    print(f"Total links: {len(all_links):,}")

    graph = build_graph(all_questions, all_answers, all_links, args.vote_threshold)

    stats = graph.stats()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['num_nodes']:,}")
    print(f"  Edges: {stats['num_edges']:,}")
    print(f"  Nodes by source:")
    for source, count in stats['nodes_by_source'].items():
        print(f"    {source}: {count:,}")
    print(f"  Edges by type:")
    for edge_type, count in stats['edges_by_type'].items():
        print(f"    {edge_type}: {count:,}")

    print(f"\nSaving to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(graph.to_dict(), f)
    print("Done!")


if __name__ == "__main__":
    main()
