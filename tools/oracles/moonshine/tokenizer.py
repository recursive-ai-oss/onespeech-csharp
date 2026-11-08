"""Exact tokenizer probes on evaluation transcripts and crafted public strings."""
import warnings
warnings.filterwarnings("ignore")
import json
from pathlib import Path
import sys
from transformers import AutoProcessor

try:
    model, clean, other, output = sys.argv[1:]
    texts = ["", " ", "  a  b\n\t", "e\u0301 café", "中文🙂", "\x00", "Hello , world ! I ' m not .", "Mr . Smith can ' t", "<s> hello</s>", "a<s>b", " a<unk> b", "<s> <<ST_0>> hi ", "<<ST_767>>", "á\u200b\u2028\u0301", "Привет мир", "العربية"]
    texts += [f"<<ST_{i}>>" for i in range(768)]
    for root in (Path(clean), Path(other)):
        entries = {}
        for file in root.rglob("*.trans.txt"):
            for line in file.read_text().splitlines():
                key, text = line.split(" ", 1)
                entries[key] = text
        texts += [entries[key] for key in sorted(entries)[:200]]
    tokenizer = AutoProcessor.from_pretrained(model, local_files_only=True).tokenizer
    rows = []
    for text in texts:
        ids = tokenizer(text)["input_ids"]
        rows.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids), "withoutSpecials": tokenizer.decode(ids, skip_special_tokens=True)})
    decode_cases = []
    for values in [[i] for i in range(256)] + [[0xC3, 0xA9], [0xE2, 0x82], [0x41, 0xFF], [0xE2, 0x28, 0xA1], [0xED, 0xA0, 0x80], [0x20], [0xF0, 0x9F, 0x99, 0x82]]:
        ids = [tokenizer.convert_tokens_to_ids(f"<0x{v:02X}>") for v in values]
        decode_cases.append({"ids": ids, "decoded": tokenizer.decode(ids), "withoutSpecials": tokenizer.decode(ids, skip_special_tokens=True)})
    Path(output).write_text(json.dumps({"oracle": "transformers5.22.2", "cases": rows, "decodeCases": decode_cases}, ensure_ascii=False))
    print(len(rows))
except Exception as error:
    print(json.dumps({"error_type": type(error).__name__, "message": str(error).splitlines()[0]}), file=sys.stderr)
    raise SystemExit(2) from None
