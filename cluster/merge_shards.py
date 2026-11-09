"""Merge sharded ASQA outputs into one ALCE-format file.
Usage: python cluster/merge_shards.py out/asqa_baseline_0_474.json out/asqa_baseline_474_948.json -o out/asqa_baseline.json
Shards must be given in order. Fails loudly on duplicate questions."""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("-o", "--output", required=True)
a = ap.parse_args()

merged = {"data": [], "args": [], "total_cost": 0.0, "azure_filter_fail": ""}
seen = set()
for path in a.shards:
    shard = json.load(open(path))
    for item in shard["data"]:
        key = item.get("sample_id", item["question"])
        if key in seen:
            raise SystemExit("duplicate example {} in {}".format(key, path))
        seen.add(key)
        merged["data"].append(item)
    merged["args"].append({"shard": path, "args": shard.get("args")})
    print("{}: {} items".format(path, len(shard["data"])))

json.dump(merged, open(a.output, "w"))
print("wrote {} items to {}".format(len(merged["data"]), a.output))
