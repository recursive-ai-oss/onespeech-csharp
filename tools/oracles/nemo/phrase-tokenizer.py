"""Black-box tokenizer fixture generation from the pinned publisher data."""
import importlib.metadata
import hashlib
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")


def main():
    if importlib.metadata.version("tokenizers") != "0.23.3":
        raise ValueError("Expected locked tokenizers 0.23.3")
    if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91":
        raise ValueError("Unexpected Parakeet tokenizer revision")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(sys.argv[1])
    texts = ["Hello", "Zorvian Plenum", "Iceland", "luminous guidance", "  Hello   world  ",
             "é e\u0301 中文 日本語 😀", "naïve façade", "\u00a0hello\tworld", "ℌ𝔢𝔩𝔩𝔬 ＡＢＣ ①", "e\u0301\u0301", "a\r\nb"]
    texts.extend(["", "  ", "▁▁a", "a▁▁b", "▁", "a  \t b", "\n", "a\u0000b", "ǅ\u00adﬁ", "e\u0301\u0301\u0301\u0301"])
    texts.extend(["term 2026 Q3: ①", "hello123world", "Ⅲ3", "42", "x 7 y", "GPT4"])
    for path in sys.argv[3:]:
        for line in Path(path).read_text().splitlines():
            if len(line) <= 1024:
                texts.append(line)
            else:
                words = line.split()
                texts.extend(" ".join(words[i:i + 32]) for i in range(0, len(words), 32))
    cases = [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids} for text in texts]
    Path(sys.argv[2]).write_text(json.dumps({"oracle": "tokenizers 0.23.3", "cases": cases}, ensure_ascii=False))
    controls = sum("<" in case["text"] or ">" in case["text"] for case in cases)
    print("Wrote", len(cases), "tokenizer cases; ordinary", len(cases) - controls, "control syntax", controls)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(type(error).__name__ + ": " + str(error)) from None
