"""Idempotent arXiv scan; every extracted statement remains a candidate."""

import argparse
import fcntl
import hashlib
import json
import re
from pathlib import Path

from arxiv_client import ArxivClient
from arxiv_feed import recent_papers
from config import DATA_DIR
from doc_indexer import MathDocIndexer
from formal_claims import check_claim
from graph_manager import GraphManager
from verification.expressions import parse_expression


PATTERN = re.compile(r"\b(open (?:problem|question|conjecture)|remains? (?:an? )?open|still open|unresolved|unproved|we conjecture|we ask whether|it is conjectured|conjecture\s+\d+(?:\.\d+)?\s*[:.]|conjecture:)", re.I)
RESOLVED = re.compile(r"\b(resolv(?:e|es|ed|ing)|prov(?:e|es|ed|ing)|confirm(?:s|ed|ing)?|settle(?:s|d)?|establish(?:es|ed)?)\b", re.I)


def extract_candidates(text):
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text)
            if PATTERN.search(sentence) and not RESOLVED.search(sentence) and 20 <= len(sentence) <= 800][:10]


def check_candidate(sentence):
    """Try the small arithmetic grammar; reject prose and unsupported mathematics."""
    expression = re.sub(r"^(?:conjecture|open problem|open question)\s*:\s*", "", sentence.strip().rstrip("."), flags=re.I)
    match = re.fullmatch(r"([a-z0-9_+*%()\s-]+)\s*(==|<=|>=|<|>)\s*([a-z0-9_+*%()\s-]+)", expression)
    if not match:
        return {"status": "UNFORMALIZED"}
    left, operator, right = match.groups()
    try:
        difference = parse_expression(left) - parse_expression(right)
        if operator == "==" and difference.simplify() == 0:
            return {"status": "CAS_IDENTITY"}
        variables = sorted(str(v) for v in difference.free_symbols)
        if not variables:
            value = difference.simplify()
            verdict = {"==": value.is_zero, "<": value.is_negative,
                       "<=": value.is_nonpositive, ">": value.is_positive,
                       ">=": value.is_nonnegative}[operator]
            return {"status": "CAS_VALID" if verdict is True else "CAS_REFUTED" if verdict is False else "UNFORMALIZED"}
        if len(variables) > 4:
            return {"status": "UNFORMALIZED"}
        result = check_claim({"variables": {v: [0, 100] for v in variables}, "assumptions": [],
                              "conclusion": f"{left} {operator} {right}"})
        return {**result, "scope": "integer variables 0..100"}
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
        return {"status": "UNFORMALIZED"}


def scan(max_papers=5, feed=recent_papers, indexer=None, graph=None, state_path=None):
    if not 1 <= max_papers <= 30:
        raise ValueError("max_papers must be 1–30")
    state_path = Path(state_path or DATA_DIR / "conjecture_scan.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(state_path.suffix + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _scan(max_papers, feed, indexer, graph, state_path)


def _scan(max_papers, feed, indexer, graph, state_path):
    state = json.loads(state_path.read_text()) if state_path.exists() else {"seen": [], "pending": [], "candidates": []}
    seen = set(state["seen"])
    pending = state.setdefault("pending", [])
    queued = {paper["id"] for paper in pending}
    pending.extend(reversed([paper for paper in feed(max_results=100) if paper["id"] not in seen and paper["id"] not in queued]))
    papers = pending[:max_papers]
    if not papers:
        return {"new_papers": 0, "new_candidates": 0, "candidates": []}
    indexer = indexer or MathDocIndexer(cache_dir=str(DATA_DIR))
    graph = graph or GraphManager(filepath=str(DATA_DIR / "graph_state.json"))
    arxiv = ArxivClient(download_dir=str(DATA_DIR / "papers"))
    added = []
    processed = set()
    failed = []
    for paper in papers:
        snippets = extract_candidates(paper["abstract"])
        try:
            name = f"arxiv_{paper['id']}"
            if name not in indexer.list_documents() or not any(node.get("text") for node in indexer.get_document_nodes(name)):
                indexer.index_document(arxiv.download_pdf(paper["id"]), doc_name=name)
            for node in indexer.get_document_nodes(name):
                snippets.extend(extract_candidates(node.get("text", "")))
        except Exception as exc:
            failed.append({"id": paper["id"], "error": str(exc)})
            continue
        if snippets:
            doc_id = f"arxiv_{paper['id']}"
            graph.add_node(doc_id, "document", paper["title"][:48], paper["title"], data=paper)
            # ponytail: save each graph finding; batch writes if scan throughput becomes a bottleneck.
            for snippet in dict.fromkeys(snippets):
                candidate_id = f"candidate_{paper['id']}_{hashlib.sha256(snippet.encode()).hexdigest()[:12]}"
                finding = {"id": candidate_id, "paper_id": paper["id"], "source": paper["abs_url"],
                           "statement": snippet, "check": check_candidate(snippet), "status": "CANDIDATE"}
                graph.add_node(candidate_id, "conjecture", snippet[:48], snippet, data=finding)
                graph.add_link(doc_id, candidate_id, label="candidate_from")
                if finding["check"]["status"] == "COUNTEREXAMPLE":
                    counterexample_id = f"counterexample_{candidate_id}"
                    graph.add_node(counterexample_id, "counterexample", "Bounded counterexample",
                                   data={"model": finding["check"]["model"], "scope": finding["check"].get("scope")})
                    graph.add_link(candidate_id, counterexample_id, label="refuted_on_bounds")
                added.append(finding)
        seen.add(paper["id"])
        processed.add(paper["id"])
    failed_ids = {paper["id"] for paper in failed}
    pending[:] = [paper for paper in pending if paper["id"] not in processed | failed_ids] + [paper for paper in papers if paper["id"] in failed_ids]
    state["seen"] = sorted(seen)
    state["candidates"].extend(added)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    temporary.replace(state_path)
    return {"new_papers": len(processed), "new_candidates": len(added), "pending_papers": len(pending),
            "index_errors": failed,
            "candidates": added}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan recent arXiv submissions for candidate open questions")
    parser.add_argument("--max-papers", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(scan(max_papers=args.max_papers), indent=2, ensure_ascii=False))
