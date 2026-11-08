"""Collect fixed slices and invoke OneSpeech's shared metric executable."""
import argparse
import json
from pathlib import Path
import subprocess


def read_rows(directory):
    return sorted((value for path in directory.glob("[0-9]*.json")
                   if "id" in (value := json.loads(path.read_text()))), key=lambda value: value["id"])


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def score(state, name, truth, ours, reference=None):
    output = state / "scores"
    output.mkdir(exist_ok=True)
    write_rows(output / (name + "-truth.jsonl"), truth)
    write_rows(output / (name + "-ours.jsonl"), ours)
    command = ["dotnet", str(state / "runner-eval/OneSpeech.Eval.dll"), "score",
               "--ref", str(output / (name + "-truth.jsonl")),
               "--hyp", str(output / (name + "-ours.jsonl")),
               "--normalizer", "en", "--resamples", "1000", "--seed", "1729",
               "--out", str(output / (name + ".json"))]
    if reference is not None:
        write_rows(output / (name + "-ref.jsonl"), reference)
        command += ["--hyp2", str(output / (name + "-ref.jsonl"))]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    (output / (name + ".log")).write_text(result.stdout)
    print(name + ": " + result.stdout.strip(), flush=True)
    return json.loads((output / (name + ".json")).read_text())


def timing_summary(report):
    rows = [row["timing"] for row in report["utterances"] if row["timing"] is not None]
    total = sum(row["timed"] for row in rows)
    value = {key: sum(row[key] for row in rows) for key in ("matched", "timed", "missing")}
    for key in ("meanStartMs", "meanEndMs", "startWithin50", "startWithin100", "endWithin50", "endWithin100"):
        value[key] = sum(row[key] * row["timed"] for row in rows if row["timed"]) / total if total else None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("wer", "timing", "long"))
    parser.add_argument("state", type=Path)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--whisper", type=Path)
    args = parser.parse_args()
    state = args.state
    if args.mode == "long":
        output = state / "scores"
        output.mkdir(exist_ok=True)
        truth_path = output / "earnings-references.jsonl"
        subprocess.run(["dotnet", str(state / "runner-eval/OneSpeech.Eval.dll"), "references",
                        "--dataset", "earnings22", "--input", str(args.root),
                        "--out", str(truth_path)], check=True)
        truth = [row for line in truth_path.read_text().splitlines()
                 if (row := json.loads(line))["id"] == "4474955"]
        assert len(truth) == 1
        for suffix in ("", "-1.7b"):
            our_path = state / "long" / ("ours" + suffix + ".json")
            ref_path = state / "long" / ("ref" + suffix + ".json")
            if not our_path.exists() or not ref_path.exists():
                print("Incomplete long-form pair " + (suffix or "-0.6b"))
                continue
            ours = json.loads(our_path.read_text())["transcript"]
            reference = json.loads(ref_path.read_text())
            score(state, "long" + (suffix or "-0.6b"), truth,
                  [dict(id="4474955", text=ours["text"])], [reference])
    elif args.mode == "wer":
        for split in ("clean", "other"):
            truth = []
            for path in (args.root / ("test-" + split)).rglob("*.trans.txt"):
                for line in path.read_text().splitlines():
                    key, text = line.split(" ", 1)
                    truth.append(dict(id=key, text=text))
            truth = sorted(truth, key=lambda row: row["id"])[:200]
            assert len(truth) == 200
            for model in ("0.6b", "1.7b"):
                for precision in ("fp32", "bf16"):
                    name = f"{model}-{precision}-{split}"
                    ours = read_rows(state / "rows" / ("ours-" + name))
                    reference = read_rows(state / "rows" / ("ref-" + name))
                    if len(ours) != 200 or len(reference) != 200:
                        print(f"Incomplete {name}: {len(ours)}/200 native; {len(reference)}/200 reference")
                        continue
                    score(state, name, truth, ours, reference)
    else:
        ours = read_rows(state / "rows/align-ours-clean")
        reference = read_rows(state / "rows/align-ref-clean")
        assert len(ours) == len(reference) == 200, "Incomplete timing slice"
        report = score(state, "align-clean", reference, ours)
        summary = timing_summary(report)
        for side, rows in (("ours", ours), ("reference", reference)):
            words = [word for row in rows for word in row["words"]]
            summary[side] = dict(utterances=len(rows), words=len(words),
                missing=sum(word.get("start") is None or word.get("end") is None for word in words))
        (state / "scores/align-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if args.whisper:
            whisper = []
            for row in ours:
                value = json.loads((args.whisper / (row["id"] + ".json")).read_text())
                transcript = value.get("transcript", value)
                words = transcript.get("words")
                if words is None:
                    words = [dict(text=word.get("text", word.get("word")), start=word["start"], end=word["end"])
                             for segment in transcript["segments"] for word in segment["words"]]
                whisper.append(dict(id=row["id"], text=transcript["text"], words=words))
            report = score(state, "align-whisper-information", whisper, ours)
            (state / "scores/align-whisper-summary.json").write_text(json.dumps(timing_summary(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
