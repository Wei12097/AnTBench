# AnTBench

Official implementation of:

**Can LLMs Spot Data Anomalies That Make Table QA Impossible?**

AnTBench evaluates whether large language models can recognize anomalies in tabular data that make table question answering unreliable or impossible. It focuses on tables with implicit inconsistencies that are objectively verifiable and relevant to the associated questions.

## Overview

Existing table reasoning benchmarks usually assume that input tables are correct. AnTBench asks:

> Can LLMs detect table anomalies and avoid producing confident answers based on corrupted data?

AnTBench contains **1,571 carefully curated instances** derived from **338 tables**, covering five verifiable anomaly categories.

The benchmark evaluates two complementary tasks:

* **Anomaly Perception**

  * Tests whether models can spontaneously recognize that a question becomes unanswerable because of table anomalies.
  * Models receive normal table QA instructions without explicit anomaly detection guidance.

* **Anomaly Detection**

  * Tests whether models can explicitly identify whether a table contains relevant anomalies.
  * Models are instructed to inspect the table for possible errors.

## Anomaly Categories

| Category | Description | Annotation label |
| --- | --- | --- |
| Calculation Error | Arithmetic inconsistencies between derived values and source values | `Calculation`, `Statistics` |
| Knowledge Conflict | Conflicts between table information and established facts | `General Knowledge` |
| Logical Conflict | Violations of deterministic logical constraints | `Rule Conflict` |
| Header Name Error | Semantic mismatch between headers and their corresponding values | `Row/Col Head Name Error` |
| Duplication Error | Duplicate identifiers or duplicated content patterns | `Row/Col Duplicate` |


## Repository Structure

```text
AnTBench/
├── data/          # README.md and all.json; download tables separately
├── evaluation/    # Metrics and evaluation
├── inference/     # API/local inference and prompts
├── scripts/       # Batch launchers
├── README.md
├── requirements.txt
└── LICENSE
```

## Installation

```bash
python -m pip install -r requirements.txt
```

Local inference requires a compatible GPU/CUDA environment.

## Data

Download the table files from [Hugging Face](https://huggingface.co/datasets/YuuChe/AnTBench) and extract them into `data/`. See [data/README.md](data/README.md) for the commands and JSON format.

## Quick Start

Run commands from the repository root. `scripts/run_experiments.sh` runs inference followed by evaluation. Its arguments are the backend (`api` or `local`) and experiment (`main` or `method`, for **Multi-Path Thinker**).

```bash
export OPENAI_API_KEY="your-api-key"
# Optional: export OPENAI_BASE_URL="your-compatible-api-endpoint"

# API: Anomaly Perception
MODELS="your-api-model" TASKS=qa bash scripts/run_experiments.sh api main

# API: Anomaly Detection
MODELS="your-api-model" TASKS=detect bash scripts/run_experiments.sh api main

# API: Multi-Path Thinker on both tasks
MODELS="your-api-model" TASKS="qa detect" \
  bash scripts/run_experiments.sh api method

# Local model: both tasks
CUDA_VISIBLE_DEVICES=0 MODELS="my-model" MODEL_PATH="/path/to/model" \
  TASKS="qa detect" bash scripts/run_experiments.sh local main

# Local model: Multi-Path Thinker
CUDA_VISIBLE_DEVICES=0 MODELS="my-model" MODEL_PATH="/path/to/model" \
  TASKS="qa detect" bash scripts/run_experiments.sh local method

# Multiple API models and all table formats
MODELS="model-a model-b" TASKS="qa detect" FORMATS="csv html latex markdown" \
  bash scripts/run_experiments.sh api main
```

`qa` uses the `Wrong` samples and evaluates refusal to answer; `detect` uses all samples and evaluates `Correct`/`Wrong` predictions. Tasks default to `qa detect`, and the table format defaults to `csv`. For local inference, set `GPU_NUM` to control parallelism or `REASONING=true` to enable the tokenizer's thinking mode where supported.

Predictions, metrics, and logs are saved under `inference/results/`, `evaluation/results/`, and `scripts/logs/`.

## Citation

If you find AnTBench useful, please cite:

```bibtex
@inproceedings{wei2026can,
  title={Can LLMs Spot Data Anomalies That Make Table QA Impossible?},
  author={Wei, Yucheng and Yang, Shenzhi and Wang, Ning and Zhang, Jinchang and Yang, Yuhang and Wang, Haobo and Zha, Liangyu and Ye, Wentao and Li, Liyao and Peng, Cheng and Chen, Gang},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  pages={21952--21987},
  year={2026}
}
```

## Contact

For questions about the benchmark, code, or data, contact weiyucheng@zju.edu.cn.

## License

This project is released under the [Apache-2.0 License](LICENSE). The third-party exact-match metric retains its copyright and license notice in `evaluation/metrics/metric/exact_match/`.
