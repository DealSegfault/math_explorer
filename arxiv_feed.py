"""Recent arXiv mathematics submissions through the documented Atom API."""

import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


ATOM = "{http://www.w3.org/2005/Atom}"
BASE = "https://export.arxiv.org/api/query"
CATEGORIES = ("math.NT", "math.AG", "math.CO")


def recent_papers(max_results=30, opener=urlopen):
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be 1–100")
    query = " OR ".join(f"cat:{category}" for category in CATEGORIES)
    url = BASE + "?" + urlencode({"search_query": query, "sortBy": "submittedDate",
                                "sortOrder": "descending", "max_results": max_results})
    request = Request(url, headers={"User-Agent": "math-solver-research/1.0 (contact: local-user)"})
    with opener(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    papers = []
    for entry in root.findall(ATOM + "entry"):
        url = (entry.findtext(ATOM + "id") or "").strip()
        paper_id = url.rsplit("/", 1)[-1].split("v", 1)[0]
        if not re.fullmatch(r"\d{4}\.\d{4,5}", paper_id):
            continue
        papers.append({
            "id": paper_id,
            "title": " ".join((entry.findtext(ATOM + "title") or "").split()),
            "abstract": " ".join((entry.findtext(ATOM + "summary") or "").split()),
            "published": entry.findtext(ATOM + "published"),
            "updated": entry.findtext(ATOM + "updated"),
            "abs_url": f"https://arxiv.org/abs/{paper_id}",
        })
    return papers
