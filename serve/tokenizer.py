# tokenizer.py - Sage-A1.5 tokenization for the serving layer.
#
# v0.1 uses HuggingFace tokenizers when available (fast, exact) and falls
# back to identity byte tokenization ONLY for smoke tests - real traffic
# requires the model's tokenizer (Qwen-family BPE, 151936 vocab).
#
# The chat template follows the Sage-A1.5 card (ChatML-compatible):
#   <|im_start|>role\n...content<|im_end|>\n   with system/knowledge sections
from __future__ import annotations

import json
import os
from typing import List, Optional

IM_START = 151644
IM_END = 151645
EOS_ID = IM_END

_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
{messages}<|im_start|>assistant
"""


class Tokenizer:
    def __init__(self, tokenizer_dir: Optional[str] = None):
        self._tok = None
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            try:
                from tokenizers import Tokenizer as HFTokenizer
                path = os.path.join(tokenizer_dir, "tokenizer.json")
                if os.path.exists(path):
                    self._tok = HFTokenizer.from_file(path)
            except ImportError:
                pass

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def encode(self, text: str) -> List[int]:
        if self._tok is None:
            # NOT for production - smoke tests only.
            return [min(ord(c), 151935) for c in text]
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        if self._tok is None:
            return "".join(chr(i) if 0 <= i < 0x110000 else "" for i in ids)
        return self._tok.decode(ids, skip_special_tokens=True)

    # ---- chat template ------------------------------------------------------
    def apply_chat(self, system: str, messages: List[dict]) -> List[int]:
        """messages: [{"role": "user"|"assistant", "content": str}, ...]"""
        turns = []
        for m in messages:
            turns.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        prompt = _TEMPLATE.format(system=system, messages="".join(turns))
        return self.encode(prompt)

    @staticmethod
    def stop_ids() -> List[int]:
        return [IM_END, EOS_ID]


def load_tokenizer_for_model(model_path: str) -> Tokenizer:
    """Look for tokenizer.json next to the GGUF, or in SAGER_TOKENIZER_DIR."""
    cand = []
    d = os.path.dirname(os.path.abspath(model_path))
    cand.append(d)
    if os.environ.get("SAGER_TOKENIZER_DIR"):
        cand.append(os.environ["SAGER_TOKENIZER_DIR"])
    for c in cand:
        if os.path.exists(os.path.join(c, "tokenizer.json")):
            return Tokenizer(c)
    return Tokenizer(None)
