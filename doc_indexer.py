import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from pageindex import md_to_tree, page_index_flash

class MathDocIndexer:
    """
    PageIndex document indexer for mathematical texts, papers, and monographs.
    Creates hierarchical tree indexes without vector databases or chunking.
    """
    def __init__(self, cache_dir: str = "/Users/mac/.gemini/antigravity/scratch/math_explorer/data"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_file = os.path.join(self.cache_dir, "index_registry.json")
        self.registry: Dict[str, Any] = self._load_registry()

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

    def index_document(self, file_path: str, doc_name: Optional[str] = None) -> str:
        """
        Indexes a Markdown or PDF document into a PageIndex tree.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if not doc_name:
            doc_name = os.path.splitext(os.path.basename(file_path))[0]

        print(f"Indexing '{file_path}' ({ext}) with PageIndex...", flush=True)

        if ext in [".md", ".markdown", ".txt"]:
            tree = asyncio.run(md_to_tree(file_path, if_add_node_text="yes"))
        elif ext == ".pdf":
            # For PDF, use page_index_flash layout tree extraction
            tree = page_index_flash(file_path, summary=False, optimize="merge")
        else:
            raise ValueError(f"Unsupported format {ext}. PageIndex supports .md, .txt, .pdf")

        # Normalize nodes
        flattened_nodes = []
        self._collect_nodes(tree.get("structure", []), flattened_nodes)

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
