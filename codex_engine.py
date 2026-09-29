import os
import subprocess
import re
from config import CODEX_BIN
from typing import Optional, Dict, Any

class CodexAstraEngine:
    """
    OpenAI Codex CLI Engine with gpt-6-astra (reasoning effort xhigh).
    Used as the frontier deep reasoning layer for ultra-complex or open math problems.
    """
    def __init__(self, codex_bin: str = CODEX_BIN, model: str = "gpt-6-astra"):
        self.codex_bin = codex_bin
        self.model = model

    def is_available(self) -> bool:
        return os.path.exists(self.codex_bin)

    def generate(self, prompt: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes Codex CLI with gpt-6-astra non-interactively.
        """
        if not self.is_available():
            raise FileNotFoundError(f"Codex binary not found at {self.codex_bin}")

        if context:
            full_prompt = (
                f"Reference Mathematical Context from Literature:\n\"\"\"\n{context}\n\"\"\"\n\n"
                f"Mathematical Task:\n{prompt}\n\n"
                f"Please provide an extremely rigorous, deep, step-by-step mathematical derivation "
                f"and proof, referencing the theorems from the context, and put the final answer in \\boxed{{}}."
            )
        else:
            full_prompt = (
                f"{prompt}\n\nPlease reason step by step with full mathematical rigor and put your final answer in \\boxed{{}}."
            )

        cmd = [
            self.codex_bin,
            "exec",
            "--skip-git-repo-check",
            "-m", self.model,
            full_prompt,
            "--ephemeral"
        ]

        print(f"\n[Invoking Codex CLI with {self.model} (reasoning effort: xhigh)]...", flush=True)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(f"Codex exited with status {result.returncode}: {result.stderr[-1000:]}")
        if not result.stdout.strip():
            raise RuntimeError("Codex returned an empty answer")
        output = result.stdout
        # Extract response text between 'codex' and 'tokens used'
        codex_match = re.search(r'\ncodex\n([\s\S]*?)(?:\ntokens used|$)', output)
        answer = codex_match.group(1).strip() if codex_match else output.strip()

        tokens_match = re.search(r'tokens used\s*([\d\s]+)', output)
        tokens_used = tokens_match.group(1).strip() if tokens_match else "unknown"

        return {
            "answer": answer,
            "raw_output": output,
            "tokens_used": tokens_used,
            "exit_code": result.returncode
        }
