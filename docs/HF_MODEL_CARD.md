---
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
tags:
  - electromagnetic-signals
  - multimodal-llm
  - iq-signals
  - low-snr
  - merlin
datasets:
  - eye1patch/EM-134K
  - eye1patch/EM-Bench
library_name: transformers
---

# MERLIN

MERLIN is a multimodal large language model framework for electromagnetic (EM) IQ signals. It connects an EM signal encoder, a projector, and a large language model to support EM signal perception and reasoning, with a focus on robustness under low Signal-to-Noise Ratio (SNR) conditions.

This is the model card draft for the MERLIN Hugging Face model repository. It can be used as the `README.md` of `eye1patch/MERLIN`.

- Paper: [MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals](https://arxiv.org/abs/2603.08174)
- Project page: [https://em-merlin.github.io/](https://em-merlin.github.io/)
- Code: `https://github.com/em-merlin/MERLIN`
- Training dataset: [eye1patch/EM-134K](https://huggingface.co/datasets/eye1patch/EM-134K)
- Evaluation benchmark: [eye1patch/EM-Bench](https://huggingface.co/datasets/eye1patch/EM-Bench)

## Model Files

The MERLIN checkpoint is released in its original format. No checkpoint format conversion is required.

Expected files may include:

```text
pytorch_model.bin
tokenizer_config.json
vocab.json
merges.txt
added_tokens.json
chat_template.jinja
config.json or model_config.yaml
```

If the checkpoint is uploaded as a project-specific PyTorch checkpoint, use the official GitHub inference code to load it.

## Base Model

MERLIN uses a large language model backbone. The public inference config currently references:

```text
Qwen/Qwen3-4B-Instruct-2507
```

Users are responsible for complying with the license and access requirements of the base model and any other third-party components.

## Usage

Install the official code repository and dependencies:

```bash
git clone https://github.com/em-merlin/MERLIN.git
cd MERLIN
conda env create -f environment.yml
conda activate EM-MLLM
```

Download this model repository, then update `configs/inference.yaml`:

```yaml
checkpoint: "/path/to/MERLIN"
em_encoder:
  model_name_or_path: "/path/to/encoder_checkpoint"
dataset_path: "/path/to/EM-Bench"
output_dir: "/path/to/outputs"
```

Run inference:

```bash
bash scripts/run_inference.sh
```

Summarize benchmark results:

```bash
python summarize_results.py --results_dir /path/to/outputs --output_file benchmark_summary.xlsx
```

## Evaluation

MERLIN is evaluated on [EM-Bench](https://huggingface.co/datasets/eye1patch/EM-Bench), a benchmark containing 4,200 expert-validated QA pairs across 14 EM tasks.

The public evaluation code computes:

- Accuracy for choice-style perception tasks.
- ROUGE-L and BLEU for open-ended reasoning tasks.

## Intended Use

MERLIN is intended for research on electromagnetic signal understanding, multimodal language models, IQ signal perception, EM reasoning, and low-SNR robustness.

## Limitations

- MERLIN is a research model and should not be used as the sole decision-maker in safety-critical, military, medical, legal, or other high-stakes settings.
- Performance may vary across signal domains, SNR distributions, sampling rates, and hardware environments.
- Users should verify all generated answers before operational use.

## License

The MERLIN code is released under Apache License 2.0. This model card declares `apache-2.0` for the MERLIN release materials. Dataset licenses and base model licenses may impose additional terms.

Please check:

- [EM-134K dataset license](https://huggingface.co/datasets/eye1patch/EM-134K)
- [EM-Bench dataset license](https://huggingface.co/datasets/eye1patch/EM-Bench)
- The base LLM license

## Citation

```bibtex
@article{shen2026merlin,
  title   = {MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals},
  author  = {Shen, Junyu and She, Zhendong and Zhang, Chenghanyu and Sun, Yuchuang and Luo, Luqing and Tan, Dingwei and Guo, Zonghao and Guo, Bo and Han, Zehua and Xie, Wupeng and Mu, Yaxin and Zhang, Peng and Li, Peipei and Wang, Fengxiang and Sun, Yangang and Sun, Maosong},
  journal = {arXiv preprint arXiv:2603.08174},
  year    = {2026}
}
```

