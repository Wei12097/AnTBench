#!/usr/bin/env bash
# Evaluate results produced by the API or local launchers without rerunning inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODELS:?Set MODELS}"
PYTHON="${PYTHON:-python}"
read -r -a model_list <<< "$MODELS"
read -r -a task_list <<< "${TASKS:-qa detect}"
read -r -a format_list <<< "${FORMATS:-csv}"
read -r -a reasoning_list <<< "${REASONING:-false}"
for reasoning in "${reasoning_list[@]}"; do
    for model in "${model_list[@]}"; do
        model_suffix="${model//\//_}"
        if [[ "$reasoning" == true ]]; then model_suffix="${model_suffix}_reasoning"; fi
        for task in "${task_list[@]}"; do
            for format in "${format_list[@]}"; do
                suffix="${BACKEND:-local}/${EXPERIMENT:-main}/$model_suffix/$task"
                score_dir="${SCORE_DIR:-evaluation/results}/$suffix"
                log_dir="${LOG_DIR:-scripts/logs}/$suffix"
                mkdir -p "$score_dir" "$log_dir"
                reference_mode=label
                if [[ "$task" == qa ]]; then reference_mode="${QA_EVAL:-refusal}"; fi
                "$PYTHON" -m evaluation.evaluate_results --reference_mode "$reference_mode" --model "$model" --input_json "${RESULT_DIR:-inference/results}/$suffix/$format.json" --task "$task" --table_format "$format" --reasoning "$reasoning" --output_csv "$score_dir/$format.csv" --output_json "$score_dir/${format}_scores.json" 2>&1 | tee "$log_dir/${format}_eval.log"
            done
        done
    done
done
