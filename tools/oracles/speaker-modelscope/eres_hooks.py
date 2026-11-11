"""Records the inputs and outputs of the named stages of the pinned ERes2NetV2 checkpoint for one utterance.

Black-box observation through torch's public forward hooks on modules named by the checkpoint's own tensor
names; no module code is read. Output: one .npz of float32 arrays keyed "<module>:in<i>" / "<module>:out".
"""
import os
import warnings
import logging
import sys
import argparse
from pathlib import Path
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

SOURCE, REVISION = "iic/speech_eres2netv2_sv_zh-cn_16k-common", "1cf80d41fb3435bd3d8df185b5c423333b2db42a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("output")
    args = parser.parse_args()
    import numpy as np
    import torch
    from modelscope.pipelines import pipeline
    pipe = pipeline(task="speaker-verification", model=os.environ.get("ERES_LOCAL", SOURCE), device="cpu") if os.environ.get("ERES_LOCAL") else pipeline(task="speaker-verification", model=SOURCE, model_revision=REVISION, device="cpu")
    modules = [(k, v) for k, v in vars(pipe).items() if isinstance(v, torch.nn.Module)]
    print("pipeline modules:", [(k, type(v).__name__) for k, v in modules], flush=True)
    root = modules[0][1].embedding_model
    print("top-level children:", [(n, type(m).__name__) for n, m in root.named_children()], flush=True)
    records = {}

    def hook(name):
        def run(module, inputs, output):
            for i, value in enumerate(inputs):
                if isinstance(value, torch.Tensor):
                    records[f"{name}:in{i}"] = value.detach().float().clone().numpy()
            if isinstance(output, torch.Tensor):
                records[f"{name}:out"] = output.detach().float().clone().numpy()
        return run

    wanted = ["conv1", "bn1", "layer1.0", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.convs.0", "layer1.0.bns.0",
              "layer1.0.convs.1", "layer1.0.bns.1", "layer1.0.conv3", "layer1.0.bn3", "layer1.0.shortcut",
              "layer2.0", "layer2.0.conv1", "layer2.0.convs.0", "layer2.0.convs.1", "layer2.0.conv3", "layer2.0.shortcut",
              "layer3.0", "layer3.0.conv1", "layer3.0.bn1", "layer3.0.bns.0", "layer3.0.bns.1", "layer3.0.bn3", "layer4.0.conv1", "layer4.0.convs.0", "layer4.0.fuse_models.0", "layer4.0", "layer3.0.convs.0", "layer3.0.convs.1", "layer3.0.fuse_models.0",
              "layer3.0.fuse_models.0.local_att", "layer3.0.conv3", "layer3.0.shortcut", "layer3", "layer4",
              "layer3_ds", "fuse34", "fuse34.local_att", "seg_1"]
    wanted += [f"layer{l}.{b}" for l, n in zip(range(1, 5), [3, 4, 6, 3]) for b in range(n)]
    named = dict(root.named_modules())
    for name in wanted:
        if name in named:
            named[name].register_forward_hook(hook(name))
        else:
            print("missing module", name, flush=True)
    with torch.inference_mode():
        value = pipe.forward(pipe.preprocess([args.audio])).detach().cpu().numpy().reshape(-1)
    records["embedding"] = value
    np.savez(args.output, **records)
    print({k: v.shape for k, v in records.items()}, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("error", type(error).__name__, str(error)[:1500])
        sys.exit(1)
