"""Adds the hyper-parameter keys the released pipeline asks for to a local copy of the pinned model's
configuration.json, one KeyError at a time. Values come from the checkpoint's tensor shapes (data):
64 stem channels, base width 26, two splits, blocks [3,4,6,3], 80 mel bins, 192-dimensional embedding."""
import json
import re
import subprocess
import sys

local, audio, output, hooks = sys.argv[1:5]
values = {"embed_dim": 192, "baseWidth": 26, "scale": 2, "expansion": 2, "m_channels": 64, "feat_dim": 80,
          "num_blocks": [3, 4, 6, 3], "pooling_func": "TSTP", "two_emb_layer": False, "block": "BasicBlockERes2NetV2",
          "block_name": "BasicBlockERes2NetV2", "stride": 1, "r": 4}
path = local + "/configuration.json"
for _ in range(20):
    result = subprocess.run([sys.executable, hooks, audio, output], capture_output=True, text=True, env=None)
    text = result.stdout + result.stderr
    match = re.search(r"error KeyError .*?\\'(\w+)\\'", text)
    if result.returncode == 0:
        print("ok")
        print(text[-3000:])
        break
    if not match or match.group(1) not in values:
        print("stuck", text[-1500:])
        break
    config = json.load(open(path))
    config["model"]["model_config"][match.group(1)] = values[match.group(1)]
    json.dump(config, open(path, "w"), indent=2)
    print("added", match.group(1), flush=True)
