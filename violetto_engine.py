import sys
import time
import torch
import threading
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class ViolettoEngine:
    """
    Limite 1B Violetto mathematical reasoning engine.
    High-throughput autoregressive model specialized in competition and research-level mathematics.
    Accelerated with native Apple Silicon MPS generation and persistent memory caching.
    """
    _shared_model = None
    _shared_tokenizer = None
    _shared_model_path = None
    _lock = threading.Lock()

    def __init__(self, model_path: str = "/Volumes/sdcard/models/limite-1b-violetto", device: str = "mps"):
        self.model_path = model_path
        self.device = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        self.dtype = torch.bfloat16 if self.device == "mps" else torch.float32

    def load(self):
        with self._lock:
            if ViolettoEngine._shared_model is not None and ViolettoEngine._shared_model_path == self.model_path:
                self.tokenizer = ViolettoEngine._shared_tokenizer
                self.model = ViolettoEngine._shared_model
                return

            print(f"Loading Limite 1B Violetto on {self.device} ({self.dtype})...", flush=True)
            t0 = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=self.dtype,
                attn_implementation="sdpa",
            ).to(self.device)
            self.model.eval()

            ViolettoEngine._shared_model = self.model
            ViolettoEngine._shared_tokenizer = self.tokenizer
            ViolettoEngine._shared_model_path = self.model_path
            print(f"Limite 1B Violetto ready for inference (loaded in {time.time() - t0:.2f}s).", flush=True)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.6,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        stream: bool = False
    ) -> str:
        """
        Generates mathematical reasoning using native PyTorch/MPS accelerated generation.
        Avoids token-by-token CPU synchronization barriers and maintains high throughput.
        """
        self.load()
        
        # Format mathematical user prompt with context if available
        if context:
            full_prompt = (
                f"Reference Mathematical Context:\n"
                f"\"\"\"\n{context}\n\"\"\"\n\n"
                f"Question / Task:\n{prompt}\n\n"
                f"Use the reference theorems and definitions to solve or prove the problem step by step."
            )
        else:
            full_prompt = prompt

        messages = [{"role": "user", "content": full_prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature <= 0.05:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(target=self._run_model_generate, kwargs={"inputs": inputs, "gen_kwargs": gen_kwargs})
            thread.start()

            full_output = ""
            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                full_output += new_text
            thread.join()
            return full_output
        else:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            # Slice off input tokens
            input_len = inputs.input_ids.shape[-1]
            generated_tokens = outputs[0, input_len:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

    def _run_model_generate(self, inputs, gen_kwargs):
        with torch.inference_mode():
            self.model.generate(**inputs, **gen_kwargs)
