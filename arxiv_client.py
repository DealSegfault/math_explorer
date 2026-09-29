import os
import re
import urllib.request
import urllib.parse
from config import PAPERS_DIR
from typing import List, Dict, Any, Optional

class ArxivClient:
    """
    Fast arXiv search, retrieval, and PDF download client.
    Integrates directly with PageIndex document tree indexing.
    """
    BASE_SEARCH_URL = "https://arxiv.org/search/"
    BASE_PDF_URL = "https://arxiv.org/pdf/"
    BASE_ABS_URL = "https://arxiv.org/abs/"

    def __init__(self, download_dir: str = str(PAPERS_DIR)):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Searches arXiv for papers matching the mathematical query.
        """
        params = urllib.parse.urlencode({
            "query": query,
            "searchtype": "all",
            "source": "header"
        })
        url = f"{self.BASE_SEARCH_URL}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                html = resp.read().decode("utf-8")
        except Exception as e:
            print(f"Error fetching arXiv search: {e}", flush=True)
            return []

        matches = re.findall(r'href=\"https://arxiv\.org/abs/(\d+\.\d+)\"', html)
        # Deduplicate while preserving order
        seen = set()
        arxiv_ids = []
        for aid in matches:
            if aid not in seen:
                seen.add(aid)
                arxiv_ids.append(aid)

        papers = []
        for aid in arxiv_ids[:max_results]:
            pos = html.find(f"arxiv.org/abs/{aid}")
            sub = html[pos:pos+2500] if pos != -1 else ""
            
            title_m = re.search(r'<p class=\"title is-5 mathjax\">([\s\S]*?)</p>', sub)
            title = re.sub(r'\s+', ' ', title_m.group(1)).strip() if title_m else f"arXiv:{aid}"
            
            abstract_m = re.search(r'<span class=\"abstract-full[\s\S]*?\">([\s\S]*?)</span>', sub)
            abstract = ""
            if abstract_m:
                abstract = re.sub(r'<.*?>', '', abstract_m.group(1))
                abstract = re.sub(r'\s+', ' ', abstract).strip()
            
            papers.append({
                "id": aid,
                "title": title,
                "abstract": abstract,
                "abs_url": f"{self.BASE_ABS_URL}{aid}",
                "pdf_url": f"{self.BASE_PDF_URL}{aid}.pdf"
            })

        return papers

    def download_pdf(self, arxiv_id: str) -> str:
        """
        Downloads the PDF of a paper by its arXiv ID.
        """
        dest_path = os.path.join(self.download_dir, f"{arxiv_id}.pdf")
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            return dest_path

        url = f"{self.BASE_PDF_URL}{arxiv_id}.pdf"
        print(f"Downloading arXiv paper {arxiv_id} from {url}...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        
        with urllib.request.urlopen(req, timeout=20) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())

        print(f"Downloaded to {dest_path} ({os.path.getsize(dest_path) // 1024} KB).", flush=True)
        return dest_path
