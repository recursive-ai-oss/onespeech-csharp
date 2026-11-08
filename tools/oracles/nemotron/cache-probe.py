"""Crafted-input public speaker-cache and postprocessor probes; no model code inspected."""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import warnings
warnings.filterwarnings("ignore")
import contextlib
import json
import sys
from pathlib import Path


def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from transformers import Nemotron3DiarizationConfig, Nemotron3DiarizationSpeakerCache, AutoProcessor
    root, out = map(Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    cfg = Nemotron3DiarizationConfig.from_pretrained(root, local_files_only=True)
    cache = Nemotron3DiarizationSpeakerCache(cfg.streaming_config, fifo_length=40, speaker_cache_update_period=300)
    summaries = []
    for step in range(2):
        prior = cache.get_embeds(torch.empty((1, 0, 512)))
        n = 0 if prior is None else prior.shape[1]
        chunk = torch.arange(step * 340 + 1, (step + 1) * 340 + 1, dtype=torch.float32)[None, :, None].expand(1, 340, 512)
        inputs = chunk if prior is None else torch.cat([prior, chunk], dim=1)
        logits = torch.full((1, inputs.shape[1] * 8, 8), -8.0)
        logits[:, :, step] = 8
        cache.update(inputs, logits, torch.zeros(512), num_chunk_frames=340)
        result = cache.get_embeds(torch.empty((1, 0, 512)))
        result[0, :, 0].numpy().astype("<f4").tofile(out / ("cache" + str(step) + ".f32"))
        summaries.append({"prior": n, "after": list(result.shape)})
    # Disable representation boosts to isolate the published frame-scoring rule.
    cfg.streaming_config.strong_boost_rate = 0.0
    cfg.streaming_config.weak_boost_rate = 0.0
    cfg.streaming_config.min_positive_scores_rate = 10.0
    cache = Nemotron3DiarizationSpeakerCache(cfg.streaming_config, fifo_length=40, speaker_cache_update_period=300)
    ids = torch.arange(1, 341, dtype=torch.float32)
    embeds = ids[None, :, None].expand(1, 340, 512)
    t = torch.arange(340, dtype=torch.float32)[:, None]
    s = torch.arange(8, dtype=torch.float32)[None, :]
    logits = 3 * torch.sin(t * .137 + s * 1.173) + 2 * torch.cos(t * .041 + s * .351)
    probs = logits.sigmoid()
    probs.numpy().astype("<f4").tofile(out / "scoring-probs.f32")
    cache.update(embeds, logits.repeat_interleave(8, dim=0)[None], torch.zeros(512), num_chunk_frames=340)
    cache.get_embeds(torch.empty((1, 0, 512)))[0, :, 0].numpy().astype("<f4").tofile(out / "scoring-selected.f32")
    default_cfg = Nemotron3DiarizationConfig.from_pretrained(root, local_files_only=True)
    default_cache = Nemotron3DiarizationSpeakerCache(default_cfg.streaming_config, fifo_length=40, speaker_cache_update_period=300)
    default_cache.update(embeds, logits.repeat_interleave(8, dim=0)[None], torch.zeros(512), num_chunk_frames=340)
    default_cache.get_embeds(torch.empty((1, 0, 512)))[0, :, 0].numpy().astype("<f4").tofile(out / "default-selected.f32")
    for speakers in (2, 4, 8):
        test_logits = logits.clone() + 2
        test_logits[:, speakers:] = -8
        test_probs = test_logits.sigmoid()
        test_probs.numpy().astype("<f4").tofile(out / ("quota-" + str(speakers) + ".probs"))
        test_cache = Nemotron3DiarizationSpeakerCache(default_cfg.streaming_config, fifo_length=40, speaker_cache_update_period=300)
        test_cache.update(embeds, test_logits.repeat_interleave(8, dim=0)[None], torch.zeros(512), num_chunk_frames=340)
        test_cache.get_embeds(torch.empty((1, 0, 512)))[0, :, 0].numpy().astype("<f4").tofile(out / ("quota-" + str(speakers) + ".selected"))
    grid = torch.full((340, 8), .001)
    grid[:32, 0] = .99
    grid[32:64, 1] = .99
    axis = (.51, .6, .7, .8, .9, .99)
    for i, a in enumerate(axis):
        for j, b in enumerate((.1, .3, .5, .7, .9, .99)):
            grid[64 + i * 6 + j, 0] = a
            grid[64 + i * 6 + j, 1] = b
    grid_logits = grid.logit()
    for label, rate in (("default", .5), ("disabled", 10.0)):
        grid_cfg = Nemotron3DiarizationConfig.from_pretrained(root, local_files_only=True)
        grid_cfg.streaming_config.min_positive_scores_rate = rate
        grid_cache = Nemotron3DiarizationSpeakerCache(grid_cfg.streaming_config, fifo_length=40, speaker_cache_update_period=300)
        grid_cache.update(embeds, grid_logits.repeat_interleave(8, dim=0)[None], torch.zeros(512), num_chunk_frames=340)
        grid_cache.get_embeds(torch.empty((1, 0, 512)))[0, :, 0].numpy().astype("<f4").tofile(out / ("grid-" + label + ".selected"))
    margin = torch.full((340, 8), .001)
    margin[:32, 0] = .99
    margin[32:64, 1] = .99
    for i, (a, b) in enumerate(((.65, .2), (.75, .3), (.85, .4), (.99, .35), (.99, .45), (.6, .3), (.9, .3))):
        margin[64 + i, 0] = a
        margin[64 + i, 1] = b
    margin_cache = Nemotron3DiarizationSpeakerCache(default_cfg.streaming_config, fifo_length=40, speaker_cache_update_period=300)
    margin_cache.update(embeds, margin.logit().repeat_interleave(8, dim=0)[None], torch.zeros(512), num_chunk_frames=340)
    margin_cache.get_embeds(torch.empty((1, 0, 512)))[0, :, 0].numpy().astype("<f4").tofile(out / "margin.selected")
    recent = torch.full((340, 8), .001)
    recent[:32, 0] = .99
    recent[32:64, 1] = .99
    for i, (a, b) in enumerate(((.7, .3), (.55, .1), (.99, .5), (.9, .45), (.51, .05), (.6, .2), (.8, .4))):
        recent[270 + i, 0] = a
        recent[270 + i, 1] = b
    recent_cache = Nemotron3DiarizationSpeakerCache(default_cfg.streaming_config, fifo_length=40, speaker_cache_update_period=300)
    recent_cache.update(embeds, recent.logit().repeat_interleave(8, dim=0)[None], torch.zeros(512), num_chunk_frames=340)
    recent_cache.get_embeds(torch.empty((1, 0, 512)))[0, :, 0].numpy().astype("<f4").tofile(out / "recency-filter.selected")
    processor = AutoProcessor.from_pretrained(root, local_files_only=True)
    for samples in (512, 1280, 1600, 52400):
        features = processor(torch.zeros(samples).numpy(), sampling_rate=16000, return_tensors="pt")
        summaries.append({"samples": samples, "features": list(features.input_features.shape), "valid": int(features.attention_mask.sum())})
    logits = torch.full((1, 64, 8), -8.0)
    logits[:, 1:2, 0] = 8
    logits[:, 4:6, 1] = 8
    logits[:, 8:16, 2] = 8
    logits[:, 18:32, 3] = 8
    summaries.append(processor.extract_speaker_dict(logits)[0])
    (out / "cache-probe.json").write_text(json.dumps(summaries))
    return summaries


if __name__ == "__main__":
    try:
        with open(os.devnull, "w") as quiet:
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                result = main()
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({"error": type(error).__name__}), file=sys.stderr)
        sys.exit(1)
