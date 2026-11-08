"""Synthetic public PyTorch attention precision observations; no model loading."""
import json
import warnings
warnings.filterwarnings("ignore")


def main():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    q = torch.randn(1, 4, 17, 16).to(torch.bfloat16)
    k = torch.randn(1, 4, 17, 16).to(torch.bfloat16)
    v = torch.randn(1, 4, 17, 16).to(torch.bfloat16)
    outputs = []
    for causal in (False, True):
        observed = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        mask = torch.zeros(17, 17)
        if causal:
            mask.masked_fill_(torch.ones(17, 17, dtype=torch.bool).triu(1), float("-inf"))
        full = (q.float() @ k.float().transpose(-1, -2)) * .25 + mask
        prob_half = full.softmax(-1).to(torch.bfloat16) @ v
        full = (full.softmax(-1) @ v.float()).to(torch.bfloat16)
        staged = (q @ k.transpose(-1, -2)) * .25 + mask.to(torch.bfloat16)
        staged = staged.float().softmax(-1).to(torch.bfloat16) @ v
        outputs.append(dict(causal=causal, elements=observed.numel(),
                            float_intermediate_mismatches=int((full != observed).sum()),
                            probability_half_mismatches=int((prob_half != observed).sum()),
                            probability_half_max_abs=float((prob_half.float() - observed.float()).abs().max()),
                            staged_half_mismatches=int((staged != observed).sum()),
                            float_max_abs=float((full.float() - observed.float()).abs().max()),
                            staged_max_abs=float((staged.float() - observed.float()).abs().max())))
    print(json.dumps(outputs))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__)
        raise SystemExit(1) from None
