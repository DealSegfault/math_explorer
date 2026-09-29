"""Pinned competition and formal-proof benchmarks; loading never calls a solver."""

import csv
import json
import re
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent / "corpus" / "benchmarks"


def load_aime(split="all", limit=None):
    """Load text-only AIME questions; 2023–24 are reserved for final evaluation."""
    if split not in {"all", "development", "evaluation"}:
        raise ValueError("split must be all, development, or evaluation")
    problems = []
    with (DATASET_DIR / "aime_1983_2024.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            year = int(row["Year"])
            if split == "development" and year >= 2023 or split == "evaluation" and year < 2023:
                continue
            answer = row["Answer"].strip()
            question = row["Question"].strip()
            # One source row accepts two answers; diagrams require a visual benchmark.
            if not re.fullmatch(r"\d{1,3}", answer) or "[asy]" in question.lower():
                continue
            problems.append({
                "id": f"AIME_{row['ID']}",
                "domain": "competition_math",
                "difficulty_tier": "aime",
                "query": question + " Give the final answer in \\boxed{}.",
                "ground_truth": str(int(answer)),
                "year": year,
                "source": "gneubig/aime-1983-2024@1f8845323b0b5994dbd74bbd0e01dc077cbcd827",
            })
            if limit is not None and len(problems) >= limit:
                break
    return problems


def load_minif2f_test(limit=None):
    """Return the 244 Lean 4 test statements, without their `sorry` placeholders."""
    problems = []
    with (DATASET_DIR / "minif2f_lean4_test.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if row.get("split") != "test" or not row["formal_statement"].rstrip().endswith(":= sorry"):
                raise ValueError("Unexpected miniF2F statement format")
            problems.append({
                "id": row["id"],
                "formal_statement": row["formal_statement"],
                "header": row["header"],
                "informal_statement": row.get("informal_stmt", ""),
                "source": "Yingjia-Wan/minif2f-lean4@e62da0c6d044a361884691dd1e4919fc8c582a68",
            })
            if limit is not None and len(problems) >= limit:
                break
    return problems
