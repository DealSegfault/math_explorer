import sys
import torch
import torch.nn.functional as F
from typing import Generator, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

class ViolettoEngine:
    """
    Limite 1B Violetto mathematical reasoning engine.
    High-throughput autoregressive model specialized in competition and research-level mathematics.
    """
    def __init__(self, model_path: str = "/Volumes/sdcard/models/limite-1b-violetto", device: str = "mps"):
        self.model_path = model_path
        self.device = device if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        self.dtype = torch.bfloat16 if self.device == "mps" else torch.float32
        
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return
            
        print(f"Loading Limite 1B Violetto on {self.device} ({self.dtype})...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        print("Limite 1B Violetto ready for inference.", flush=True)

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.6,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        stream: bool = True
    ) -> str:
        """
        Generates mathematical reasoning and solution.
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

        input_ids = inputs.input_ids
        past_key_values = None
        generated_tokens = []
        full_output = ""

        for step in range(max_tokens):
            with torch.inference_mode():
                if past_key_values is None:
                    outputs = self.model(input_ids=input_ids, use_cache=True)
                else:
                    outputs = self.model(input_ids=input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :].float().clone()

                # Repetition penalty
                if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                    for token_id in set(generated_tokens[-64:]):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

                # Top-k with temperature sampling
                top_logits, top_indices = torch.topk(logits, k=top_k, dim=-1)
                probs = F.softmax(top_logits / temperature, dim=-1)
                
                sample_idx = torch.multinomial(probs.cpu(), num_samples=1).to(self.device)
                next_token = top_indices.gather(-1, sample_idx)
                token_id = next_token[0].item()

                if token_id == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(token_id)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                
                if stream:
                    sys.stdout.write(token_str)
                    sys.stdout.flush()
                    
                full_output += token_str

        return full_output
