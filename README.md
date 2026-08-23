# AnTBench

Official implementation of:

**Can LLMs Spot Data Anomalies That Make Table QA Impossible?**

AnTBench is a benchmark for evaluating whether large language models can recognize anomalies in tabular data that make table question answering unreliable or impossible. AnTBench focuses on realistic scenarios where tables contain hidden inconsistencies and models need to decide whether they can trust the input data.

The code is currently being organized. Full dataset and evaluation instructions will be released soon.


## Overview

Existing table reasoning benchmarks usually evaluate models under the assumption that input tables are correct. AnTBench studies a different question:

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

AnTBench includes five types of table anomalies:

* **Calculation Error**

  * Arithmetic inconsistencies between derived values and their source values.

* **Knowledge Conflict**

  * Conflicts between table information and established real-world facts.

* **Logical Conflict**

  * Violations of deterministic logical constraints.

* **Header Name Error**

  * Semantic mismatch between column headers and their corresponding values.

* **Duplication Error**

  * Duplicate identifiers or duplicated content patterns.

These anomaly types are designed to be implicit, objectively verifiable, and relevant to the associated questions.

## Repository Structure

```text
AnTBench/
├── data/
│   ├── tables/
│   └── questions/
├── evaluation/
│   ├── perception/
│   └── detection/
├── baselines/
├── scripts/
├── configs/
└── README.md
```

## Installation

```bash

```

## Quick Start

### Evaluate Anomaly Perception

```bash

```

### Evaluate Anomaly Detection

```bash

```

## Evaluation Metrics

* **Anomaly Perception**

  * Exact Match (EM) for refusal-based evaluation.

* **Anomaly Detection**

  * F1 score on the anomaly class.

## Citation

If you find AnTBench useful, please cite:

```bibtex
@article{antbench2026,
  title={Can LLMs Spot Data Anomalies That Make Table QA Impossible?},
  author={Anonymous},
  journal={EMNLP},
  year={2026}
}
```

## License

This project is released under the Apache-2.0 License.
