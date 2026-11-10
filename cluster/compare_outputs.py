"""Check that two ASQA runs generated the same answers and differ only in formatting.

Used for the --keep_special_tokens finding: greedy decoding and token-id based critique scores mean the
generations should be identical; only the decoded text (and therefore the citation markers) should change.

  sbatch --partition=MGPU-TC2 --qos=normal --gres=gpu:1 --cpus-per-task=1 --mem=4G --time=00:10:00 \
    --output=logs/compare_%j.log \
    --wrap="python3 cluster/compare_outputs.py out/asqa_baseline_0_end.json out/asqa_keeptok_0_end.json"

Standard library only.
"""
import json, re, sys

REFLECTION = re.compile(r"\[(?:Fully supported|Partially supported|No support / Contradictory|No Retrieval|Retrieval|"
                        r"Irrelevant|Relevant|Continue to Use Evidence|Utility:[1-5])\]|</s>|</?paragraph>")


def answer_text(item):
    """Output with citation markers and leftover tokens removed, letters and digits only."""
    s = re.sub(r"\[\d+\]", " ", item.get("output", ""))
    s = REFLECTION.sub(" ", s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def generation(item):
    """Raw generated segments with reflection tokens removed (what the model actually wrote)."""
    segs = item.get("intermediate") or []
    return [re.sub(r"\s+", " ", REFLECTION.sub("", s)).strip() for s in segs]


def main(a_path, b_path):
    a = json.load(open(a_path))["data"]
    b = json.load(open(b_path))["data"]
    assert len(a) == len(b), (len(a), len(b))
    same_q = sum(x["question"] == y["question"] for x, y in zip(a, b))
    same_gen = sum(generation(x) == generation(y) for x, y in zip(a, b))
    same_ans = sum(answer_text(x) == answer_text(y) for x, y in zip(a, b))
    multi_a = sum(len([s for s in generation(x) if s]) > 1 for x in a)
    multi_b = sum(len([s for s in generation(y) if s]) > 1 for y in b)
    cites = lambda d: sum(len(re.findall(r"\[\d+\]", x.get("output", ""))) for x in d) / len(d)
    print("items:", len(a), "| same question order:", same_q)
    print("identical generations (reflection tokens removed):", same_gen)
    print("identical answer text (citations removed):", same_ans)
    print("answers with >1 segment: A {} | B {}".format(multi_a, multi_b))
    print("mean citation markers per answer: A {:.2f} | B {:.2f}".format(cites(a), cites(b)))
    shown = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if answer_text(x) != answer_text(y) and shown < 3:
            print("\n--- differs at item", i, x["question"])
            print("A:", x.get("output", "")[:300])
            print("B:", y.get("output", "")[:300])
            shown += 1


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
