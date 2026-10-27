# AnTBench Data

## Download

Annotations are provided in `all.json`. Download the table files from [Hugging Face](https://huggingface.co/datasets/YuuChe/AnTBench). From the repository root:

```bash
curl -L --fail \
  https://huggingface.co/datasets/YuuChe/AnTBench/resolve/main/AnTBench.zip \
  -o /tmp/AnTBench.zip
unzip -n /tmp/AnTBench.zip -d data
```

The archive contains only `csv/`, `html/`, `latex/`, and `markdown/`. A table named `employment-1-1` is stored as `data/csv/employment-1-1.csv`, with corresponding files in the other formats. `all.json` remains in this code repository.

## JSON Format

`all.json` is a JSON array. Each record contains:

```json
{
  "id": 1,
  "FileName": "employment-1-1",
  "Question": "Question associated with the table.",
  "ErrorType": "Calculation",
  "ErrorLocation": "13,7",
  "Label": "Wrong"
}
```

| Field | Description |
| --- | --- |
| `id` | Unique sample identifier |
| `FileName` | Table filename without extension, shared by all formats |
| `Question` | Question associated with the table |
| `ErrorType` | Anomaly category; `""` for `Correct` samples |
| `ErrorLocation` | Anomaly location; `""` for `Correct` samples |
| `Label` | `Correct` or `Wrong` |
