"""Persistent Lean 4 REPL worker. A proof is checked against its exact statement."""

import json
import os
import re
import select
import shlex
import shutil
import subprocess
import threading
import time
import textwrap


class LeanWorker:
    def __init__(self, command=None, timeout=120):
        configured = command or os.getenv("MATH_LEAN_REPL_CMD")
        self.command = shlex.split(configured) if isinstance(configured, str) else configured
        if not self.command and shutil.which("repl"):
            self.command = ["repl"]
        self.workdir = os.getenv("MATH_LEAN_WORKDIR")
        self.timeout = timeout
        self.process = None
        self.lock = threading.Lock()
        self.pending = b""

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def _request(self, payload):
        if not self.command:
            return {"error": "Lean REPL unavailable; set MATH_LEAN_REPL_CMD"}
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.process = subprocess.Popen(self.command, cwd=self.workdir,
                                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, bufsize=0)
                self.pending = b""
            try:
                self.process.stdin.write(json.dumps(payload).encode() + b"\n\n")
                self.process.stdin.flush()
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline:
                    while b"\n\n" in self.pending:
                        block, self.pending = self.pending.split(b"\n\n", 1)
                        block = block.strip()
                        if not block:
                            continue
                        try:
                            return json.loads(block)
                        except json.JSONDecodeError:
                            if block.startswith(b"Error:"):
                                return {"error": block.decode(errors="replace")}
                    ready, _, _ = select.select([self.process.stdout], [], [], max(0, deadline - time.monotonic()))
                    if ready:
                        chunk = os.read(self.process.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        self.pending += chunk
                self.close()
                return {"error": "Lean REPL timed out or exited"}
            except (BrokenPipeError, OSError) as exc:
                self.close()
                return {"error": str(exc)}

    def prove(self, formal_statement, proof, header="import Init"):
        """Replace only the final `sorry` in a known theorem template."""
        if (not formal_statement.lstrip().startswith("theorem ")
                or not formal_statement.rstrip().endswith(":= sorry")
                or formal_statement.count(":=") != 1
                or len(re.findall(r"\btheorem\b", formal_statement)) != 1
                or re.search(r"\b(axiom|unsafe|sorry|admit)\b", formal_statement[:-len(":= sorry")])
                or re.search(r"(?m)^\s*(?:def|lemma|example|set_option|namespace|end|attribute|macro|open|import)\b", formal_statement)):
            return {"status": "INVALID", "reason": "Expected a theorem ending in := sorry"}
        if re.search(r"\b(sorry|admit|axiom|unsafe)\b", proof) or re.search(r"\b(axiom|unsafe)\b", header):
            return {"status": "INVALID", "reason": "Untrusted proof escape or axiom"}
        if any(line.strip() and not re.match(r"^(import|open|open scoped)\s+", line.strip()) for line in header.splitlines()):
            return {"status": "INVALID", "reason": "Header may only import or open modules"}
        name = re.search(r"\btheorem\s+([A-Za-z_][A-Za-z_0-9']*)", formal_statement).group(1)
        statement = formal_statement.rstrip()[:-len(":= sorry")] + ":= by\n" + textwrap.indent(proof.strip(), "  ")
        response = self._request({"cmd": header + "\n" + statement})
        if "error" in response:
            return {"status": "UNAVAILABLE", "reason": response["error"]}
        messages = response.get("messages", [])
        if response.get("sorries") or any(m.get("severity") in {"error", "warning"} for m in messages):
            return {"status": "REJECTED", "messages": messages, "sorries": response.get("sorries", [])}
        if "env" not in response:
            return {"status": "REJECTED", "reason": "REPL did not return an environment", "response": response}
        axioms = self._request({"cmd": f"#print axioms {name}", "env": response["env"]})
        if "error" in axioms or "sorryAx" in json.dumps(axioms):
            return {"status": "REJECTED", "reason": "Axiom inspection failed or proof uses sorryAx", "axioms": axioms}
        return {"status": "PROVED", "theorem": name, "axioms": axioms.get("messages", []),
                "formal_statement": formal_statement, "checked_source": header + "\n" + statement}


if __name__ == "__main__":
    worker = LeanWorker()
    try:
        print(json.dumps(worker.prove("theorem one_add_one : (1 : Nat) + 1 = 2 := sorry", "rfl"), indent=2))
    finally:
        worker.close()
