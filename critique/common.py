"""Small helpers shared by the critique scripts (no GPU, no vLLM, only the standard library)."""
import json
import re

# Copied from self-rag/retrieval_lm/utils.py so these scripts run without vLLM installed.
SUPPORT_TOKENS = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
REFLECTION_TOKENS = SUPPORT_TOKENS + ["[No Retrieval]", "[Retrieval]", "[Irrelevant]", "[Relevant]", "<paragraph>",
                                      "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]",
                                      "[Utility:5]"]


def postprocess(pred):
    """Same as self-rag/retrieval_lm/utils.py postprocess (keeps [Continue to Use Evidence])."""
    for item in REFLECTION_TOKENS:
        pred = pred.replace(item, "")
    pred = pred.replace("</s>", "")
    if len(pred) == 0:
        return ""
    if pred[0] == " ":
        pred = pred[1:]
    return pred


def fix_spacing(input_text):
    """Same as self-rag/retrieval_lm/utils.py fix_spacing."""
    return re.sub(r'(?<=\w)([.!?])(?=\w)', r'\1 ', input_text)


def clean_claim(text):
    """Plain sentence text: reflection tokens and [Continue to Use Evidence] removed, spaces tidied."""
    text = postprocess(text).replace("[Continue to Use Evidence]", " ")
    return " ".join(text.split())


def load_split_indices(paths):
    """Comma-separated list of split files (critique/splits/asqa_A.json, ...) -> sorted list of indices."""
    indices = []
    for path in [p for p in paths.split(",") if p]:
        indices += json.load(open(path))["indices"]
    assert len(indices) == len(set(indices)), "the split files overlap"
    return sorted(indices)


def load_outputs(path):
    """ALCE-format output file -> {question index: item}. Uses the q_idx field when present, else position."""
    data = json.load(open(path))["data"]
    return {item.get("q_idx", pos): item for pos, item in enumerate(data)}
