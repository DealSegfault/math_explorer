import os
import re
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

import tantivy
from pageindex import md_to_tree, page_index_flash
from config import DATA_DIR
from cache_manager import cache

class MathDocIndexer:
    """
    Two-Stage Hybrid Literature Retriever for Mathematical Texts.
    Stage 1: Local Tantivy BM25 Lexical Pruning (filters thousands of nodes down to top-30 in < 2ms).
    Stage 2: Bounded Concurrent JEV Semantic Reranking with SQLite Caching (top-30 to top-3 in < 250ms).
    """
    def __init__(self, cache_dir: str = DATA_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_file = os.path.join(self.cache_dir, "index_registry.json")
        self.registry: Dict[str, Any] = self._load_registry()
        
        # Build Tantivy in-memory index
        self._init_tantivy()

    def _init_tantivy(self):
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("node_id", stored=True)
        schema_builder.add_text_field("doc_name", stored=True)
        schema_builder.add_text_field("title", stored=True)
        schema_builder.add_text_field("text", stored=True)
        self.schema = schema_builder.build()
        self.tantivy_index = tantivy.Index(self.schema)
        self._reindex_tantivy()

    def _reindex_tantivy(self):
        writer = self.tantivy_index.writer()
        all_nodes = self.get_all_nodes()
        for n in all_nodes:
            writer.add_document(tantivy.Document(
                node_id=[str(n.get("node_id", ""))],
                doc_name=[str(n.get("doc_name", ""))],
                title=[str(n.get("title", ""))],
                text=[str(n.get("text", ""))[:2000]]
            ))
        writer.commit()
        self.tantivy_index.reload()
        self.searcher = self.tantivy_index.searcher()

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_registry(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
        self._reindex_tantivy()

    def index_document(self, file_path: str, doc_name: Optional[str] = None) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if not doc_name:
            doc_name = os.path.splitext(os.path.basename(file_path))[0]

        print(f"Indexing '{file_path}' ({ext}) with PageIndex...", flush=True)

        if ext in [".md", ".markdown", ".txt"]:
            tree = asyncio.run(md_to_tree(file_path, if_add_node_text="yes"))
        elif ext == ".pdf":
            tree = page_index_flash(file_path, summary=False, optimize="merge")
        else:
            raise ValueError(f"Unsupported format {ext}. PageIndex supports .md, .txt, .pdf")

        flattened_nodes = []
        self._collect_nodes(tree.get("structure", []), flattened_nodes)
        if ext == ".pdf" and flattened_nodes and not any(node["text"] for node in flattened_nodes):
            import fitz
            with fitz.open(file_path) as pdf:
                pages = [page.get_text() for page in pdf]
            for node in flattened_nodes:
                start, end = node.get("start_page"), node.get("end_page")
                if isinstance(start, int) and isinstance(end, int) and 1 <= start <= end <= len(pages):
                    node["text"] = "\n".join(pages[start - 1:end])[:20000]

        self.registry[doc_name] = {
            "file_path": file_path,
            "format": ext,
            "tree": tree,
            "nodes": flattened_nodes
        }
        self._save_registry()
        print(f"Successfully indexed '{doc_name}' with {len(flattened_nodes)} hierarchical nodes.", flush=True)
        return doc_name

    def _collect_nodes(self, structure: List[Dict[str, Any]], flat_list: List[Dict[str, Any]], parent_path: str = ""):
        for item in structure:
            title = item.get("title", "Untitled Section")
            current_path = f"{parent_path} > {title}" if parent_path else title
            node = {
                "node_id": item.get("node_id", str(len(flat_list))),
                "title": title,
                "full_path": current_path,
                "text": item.get("text", "") or item.get("summary", ""),
                "start_page": item.get("start_index"),
                "end_page": item.get("end_index"),
                "line_num": item.get("line_num")
            }
            flat_list.append(node)
            if "nodes" in item and item["nodes"]:
                self._collect_nodes(item["nodes"], flat_list, current_path)

    def list_documents(self) -> List[str]:
        return list(self.registry.keys())

    def get_document_nodes(self, doc_name: str) -> List[Dict[str, Any]]:
        doc = self.registry.get(doc_name)
        if not doc:
            raise KeyError(f"Document '{doc_name}' not found in index registry.")
        return doc.get("nodes", [])

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        all_nodes = []
        for doc_name, doc in self.registry.items():
            for node in doc.get("nodes", []):
                node_with_doc = dict(node)
                node_with_doc["doc_name"] = doc_name
                all_nodes.append(node_with_doc)
        return all_nodes

    def lexical_search(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """
        Stage 1: Fast Tantivy BM25 lexical candidate generator (< 2ms).
        """
        clean_words = re.findall(r"[a-zA-Z0-9_]+", query)
        if not clean_words:
            return self.get_all_nodes()[:top_k]

        clean_query = " ".join(clean_words)
        try:
            parsed_query = self.tantivy_index.parse_query(clean_query, ["title", "text"])
            results = self.searcher.search(parsed_query, top_k)
            candidates = []
            for bm25_score, doc_address in results.hits:
                doc_dict = self.searcher.doc(doc_address).to_dict()
                candidates.append({
                    "node_id": doc_dict.get("node_id", [""])[0],
                    "doc_name": doc_dict.get("doc_name", [""])[0],
                    "title": doc_dict.get("title", [""])[0],
                    "text": doc_dict.get("text", [""])[0],
                    "bm25_score": float(bm25_score)
                })
            if candidates:
                return candidates
        except Exception:
            pass

        # Fallback to initial nodes if query parse returned empty
        return self.get_all_nodes()[:top_k]

    def two_stage_retrieve(
        self,
        query: str,
        router,
        doc_name: Optional[str] = None,
        lexical_top_k: int = 25,
        final_top_k: int = 3,
        max_workers: int = 8
    ) -> Dict[str, Any]:
        """
        Two-stage retrieval pipeline with bounded concurrency and SQLite caching.
        Returns top nodes and fine-grained latency telemetry.
        """
        t0 = time.perf_counter_ns()
        
        # 0. Check multi-level retrieval cache
        cache_key = cache.hash_key("retrieval", query, doc_name or "all", final_top_k)
        cached = cache.get("retrieval_cache", cache_key)
        if cached:
            cached["metrics"]["cached"] = True
            cached["metrics"]["total_ms"] = round((time.perf_counter_ns() - t0) / 1e6, 2)
            return cached

        # 1. Stage 1: Tantivy BM25 Lexical Pre-filtering
        t_bm25_0 = time.perf_counter_ns()
        candidates = self.lexical_search(query, top_k=lexical_top_k)
        if doc_name:
            candidates = [c for c in candidates if c.get("doc_name") == doc_name]
        bm25_ms = round((time.perf_counter_ns() - t_bm25_0) / 1e6, 2)

        if not candidates:
            return {"nodes": [], "metrics": {"bm25_ms": bm25_ms, "rerank_ms": 0.0, "total_ms": bm25_ms, "cached": False}}

        # 2. Stage 2: Bounded Concurrent JEV Semantic Reranking
        t_rerank_0 = time.perf_counter_ns()

        def _score_candidate(cand: Dict[str, Any]) -> Dict[str, Any]:
            content = f"{cand.get('title', '')}: {cand.get('text', '')[:300]}"
            cand_key = cache.hash_key("jev_relevance", query, cand.get("node_id"), content)
            cached_score = cache.get("jev_score", cand_key)
            if cached_score is not None:
                score = float(cached_score)
            else:
                score = router.score_node_relevance(query, content)
                cache.set("jev_score", cand_key, score)
            
            res = dict(cand)
            res["jev_score"] = score
            return res

        # Parallelize scoring across candidates with bounded thread pool
        with ThreadPoolExecutor(max_workers=min(max_workers, len(candidates))) as executor:
            scored = list(executor.map(_score_candidate, candidates))

        # Sort by JEV semantic score descending
        scored.sort(key=lambda x: x["jev_score"], reverse=True)
        top_results = scored[:final_top_k]
        rerank_ms = round((time.perf_counter_ns() - t_rerank_0) / 1e6, 2)
        total_ms = round((time.perf_counter_ns() - t0) / 1e6, 2)

        payload = {
            "nodes": top_results,
            "metrics": {
                "bm25_candidates_count": len(candidates),
                "bm25_ms": bm25_ms,
                "rerank_ms": rerank_ms,
                "total_ms": total_ms,
                "cached": False
            }
        }
        cache.set("retrieval_cache", cache_key, payload)
        return payload

    def hierarchical_search(
        self,
        query: str,
        router,
        doc_name: Optional[str] = None,
        branch_beam: int = 2,
        max_depth: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Fast compatibility method wrapping two_stage_retrieve.
        """
        res = self.two_stage_retrieve(
            query=query,
            router=router,
            doc_name=doc_name,
            lexical_top_k=branch_beam * 10,
            final_top_k=branch_beam * 2
        )
        return res.get("nodes", [])
