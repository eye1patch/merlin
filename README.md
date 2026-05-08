# MERLIN

Official repository for **MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals**, accepted to CVPR 2026.

MERLIN is a multimodal large language model framework for electromagnetic (EM) IQ signals. It connects a signal encoder, a projection module, and a large language model to support EM signal perception and reasoning, with a focus on robustness under low Signal-to-Noise Ratio (SNR) conditions.

- Paper: [arXiv:2603.08174](https://arxiv.org/abs/2603.08174)
- Project page: [em-merlin.github.io](https://em-merlin.github.io/)
- Training dataset: [eye1patch/EM-134K](https://huggingface.co/datasets/eye1patch/EM-134K)
- Evaluation benchmark: [eye1patch/EM-Bench](https://huggingface.co/datasets/eye1patch/EM-Bench)
- Model weights: [eye1patch/MERLIN](https://huggingface.co/eye1patch/MERLIN)

## Release Scope

This repository is an evaluation and inference release. It includes the model definition, data loading utilities, inference scripts, batch inference utilities, and benchmark summarization code.

Training code, training-only configs, private data paths, intermediate checkpoints, and experiment logs are intentionally excluded from this public repository.

## Repository Structure

```text
.
|-- batch_inference.py          # Batch inference over multiple checkpoints
|-- inference.py                # Single evaluation/inference entry point
|-- summarize_results.py        # EM-Bench result aggregation
|-- configs/                    # Public inference and batch inference configs
|-- data/                       # Dataset wrappers and collators for evaluation
|-- models/                     # MERLIN model components
|-- scripts/                    # Public inference scripts
|-- utils/                      # Utility functions and constants
`-- docs/                       # Release notes and model card draft
```

## Installation

Create the environment from the provided Conda file:

```bash
conda env create -f environment.yml
conda activate EM-MLLM
```

Alternatively, install the Python dependencies manually:

```bash
pip install -r requirements.txt
```

The released environment was prepared for PyTorch with CUDA support. Please adjust the PyTorch installation command if your CUDA version differs.

## Model Weights

MERLIN weights are planned to be hosted on Hugging Face Model Hub:

```text
https://huggingface.co/eye1patch/MERLIN
```

The weight files will be uploaded in their original checkpoint format. No conversion to another format is required. After downloading the model files, update the relevant paths in `configs/inference.yaml`, especially:

- `checkpoint`
- `em_encoder.model_name_or_path`
- `dataset_path`
- `output_dir`

If the released checkpoint depends on an external base LLM, such as Qwen, please make sure you have accepted and followed the base model license and access requirements.

## Data

The public datasets are available on Hugging Face:

- [EM-134K](https://huggingface.co/datasets/eye1patch/EM-134K): large-scale EM signal-text instruction dataset.
- [EM-Bench](https://huggingface.co/datasets/eye1patch/EM-Bench): evaluation benchmark with 4,200 QA pairs across 14 EM tasks.

For local inference, download or prepare EM-Bench in a Hugging Face `datasets` disk format compatible with `datasets.load_from_disk`, then point `dataset_path` in `configs/inference.yaml` to that local directory.

## Inference

Edit `configs/inference.yaml` to match your local model and dataset paths, then run:

```bash
bash scripts/run_inference.sh
```

The script launches:

```bash
accelerate launch --multi_gpu --mixed_precision=bf16 inference.py --config configs/inference.yaml
```

Generated JSON files are written to the configured `output_dir`.

## Batch Inference

For evaluating multiple checkpoints or model variants, edit `configs/batch_inference.yaml`, then run:

```bash
bash scripts/run_batch_inference.sh
```

## Evaluation Summary

After inference, summarize EM-Bench results with:

```bash
python summarize_results.py --results_dir /path/to/inference/results --output_file benchmark_summary.xlsx
```

The script computes accuracy for choice-style tasks and ROUGE-L/BLEU for open-ended tasks, then exports an Excel summary.

## License

The code in this repository is released under the Apache License 2.0. Dataset and base-model licenses may impose additional terms. Please check the corresponding Hugging Face pages before redistribution or commercial use.

## Citation

If you find MERLIN useful, please cite:

```bibtex
@article{shen2026merlin,
  title   = {MERLIN: Building Low-SNR Robust Multimodal LLMs for Electromagnetic Signals},
  author  = {Shen, Junyu and She, Zhendong and Zhang, Chenghanyu and Sun, Yuchuang and Luo, Luqing and Tan, Dingwei and Guo, Zonghao and Guo, Bo and Han, Zehua and Xie, Wupeng and Mu, Yaxin and Zhang, Peng and Li, Peipei and Wang, Fengxiang and Sun, Yangang and Sun, Maosong},
  journal = {arXiv preprint arXiv:2603.08174},
  year    = {2026}
}
```

