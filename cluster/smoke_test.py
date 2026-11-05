"""Smoke test: the README quick-start prompts through vLLM on one GPU.
Expected (from the upstream README):
  1) "The capital of France is Paris.[Utility:5]</s>"
  2) "[Relevant]Alpacas are considerably smaller than llamas.[Fully supported][Utility:5]</s>"
Passes when prompt 1 ends in a [Utility:N] token and prompt 2 has a relevance token and a [Utility:N] token."""
import os, sys
from vllm import LLM, SamplingParams

model_name = os.environ.get("SELFRAG_MODEL", "selfrag/selfrag_llama2_7b")
model = LLM(model_name, download_dir=os.environ.get("HF_HOME"), dtype="half")
sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)

def fmt(q, paragraph=None):
    p = "### Instruction:\n{0}\n\n### Response:\n".format(q)
    if paragraph:
        p += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return p

q1 = "What is the capital of France?"
q2 = "Can you tell me the difference between llamas and alpacas?"
para = ("The alpaca (Lama pacos) is a species of South American camelid mammal. It is similar to, "
        "and often confused with, the llama. Alpacas are considerably smaller than llamas, and unlike "
        "llamas, they were not bred to be working animals, but were bred specifically for their fiber.")
prompts = [fmt(q1), fmt(q2, para)]
outs = model.generate(prompts, sp)
texts = [o.outputs[0].text for o in outs]
for p, t in zip(prompts, texts):
    print("PROMPT:", p.replace("\n", " ")[:120])
    print("OUTPUT:", t)
    print("-" * 60)

ok = ("[Utility:" in texts[0]) and \
     (("[Relevant]" in texts[1]) or ("[Irrelevant]" in texts[1])) and ("[Utility:" in texts[1])
print("SMOKE_OK" if ok else "SMOKE_FAIL")
sys.exit(0 if ok else 1)
