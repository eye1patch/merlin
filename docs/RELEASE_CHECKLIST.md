# MERLIN Release Checklist

Use this checklist before pushing the public GitHub repository or publishing model weights.

## GitHub Code Release

- [ ] Confirm this is an evaluation-only release.
- [ ] Confirm `training/` is not staged.
- [ ] Confirm `scripts/train*.sh`, `scripts/debug*.sh`, and other private experiment scripts are not staged.
- [ ] Confirm training-only configs such as `configs/stage*_config.yaml` and `configs/ds_config.json` are not staged.
- [ ] Confirm no private datasets, local data caches, logs, wandb runs, tensorboard logs, or output folders are staged.
- [ ] Confirm no intermediate checkpoint directories are staged.
- [ ] Confirm no large weight files such as `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, or `*.safetensors` are staged for GitHub.
- [ ] Confirm public inference configs do not expose private paths, credentials, tokens, or machine-specific storage locations.
- [ ] Confirm `README.md`, `LICENSE`, and `CITATION.cff` are present.
- [ ] Confirm the README links are valid:
  - Paper: `https://arxiv.org/abs/2603.08174`
  - Project page: `https://em-merlin.github.io/`
  - EM-134K: `https://huggingface.co/datasets/eye1patch/EM-134K`
  - EM-Bench: `https://huggingface.co/datasets/eye1patch/EM-Bench`
  - Model weights: `https://huggingface.co/eye1patch/MERLIN`

## Suggested Git Commands

From the repository root:

```bash
git init
git add .
git status
```

Before committing, manually inspect the staged files:

```bash
git diff --cached --name-only
```

The staged list should contain only public code, public configs, documentation, and dependency files.

## Hugging Face Model Release

- [ ] Create the model repository, for example `eye1patch/MERLIN`.
- [ ] Upload the original checkpoint files without converting format.
- [ ] Include tokenizer files required by the LLM.
- [ ] Include any MERLIN-specific config needed by `inference.py`.
- [ ] State whether the uploaded checkpoint is a full model, a merged model, or a project-specific checkpoint.
- [ ] State the required base LLM and its license if users need to download it separately.
- [ ] Include a model card with links to the paper, project page, GitHub code, EM-134K, and EM-Bench.
- [ ] Confirm the weight license and base model license are compatible with redistribution.

## Release Notes

Recommended GitHub release title:

```text
v1.0.0 - Evaluation Code Release
```

Recommended release note:

```text
This release contains the official MERLIN evaluation and inference code for EM-Bench. Training code and training-only assets are intentionally excluded. Model weights are hosted on Hugging Face Model Hub.
```

