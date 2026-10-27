#!/usr/bin/env bash
# Run AnTBench inference and evaluation across models, tasks, and table formats.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
backend="${1:?backend required}"
experiment="${2:?experiment required}"
: "${MODELS:?Set MODELS to one or more space-separated model names}"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data}"
read -r -a model_list <<< "$MODELS"
read -r -a task_list <<< "${TASKS:-qa detect}"
read -r -a format_list <<< "${FORMATS:-csv}"
read -r -a reasoning_list <<< "${REASONING:-false}"
if [[ "$backend" == api ]]; then
    : "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
    reasoning_list=(false)
fi
if [[ "$experiment" != main && "$experiment" != method ]]; then
    echo 'EXPERIMENT must be main or method' >&2
    exit 2
fi
if [[ -n "${MODEL_PATH:-}" && ${#model_list[@]} -ne 1 ]]; then
    echo 'MODEL_PATH requires a single model in MODELS' >&2
    exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a gpu_array <<< "$CUDA_VISIBLE_DEVICES"
    gpu_num="${GPU_NUM:-${#gpu_array[@]}}"
else
    gpu_num="${GPU_NUM:-1}"
fi
for reasoning in "${reasoning_list[@]}"; do
    for model in "${model_list[@]}"; do
        model_suffix="${model//\//_}"
        if [[ "$reasoning" == true ]]; then model_suffix="${model_suffix}_reasoning"; fi
        for task in "${task_list[@]}"; do
            case "$task" in
                qa) input_json="${INPUT_JSON:-${QA_INPUT:-$DATA_DIR/all.json}}" ;;
                detect) input_json="${INPUT_JSON:-${DETECT_INPUT:-$DATA_DIR/all.json}}" ;;
                *) echo "Unknown task: $task" >&2; exit 2 ;;
            esac
            [[ -f "$input_json" ]] || { echo "Missing dataset: $input_json. See data/README.md" >&2; exit 2; }
            for format in "${format_list[@]}"; do
                result_dir="${RESULT_DIR:-inference/results}/$backend/$experiment/$model_suffix/$task"
                score_dir="${SCORE_DIR:-evaluation/results}/$backend/$experiment/$model_suffix/$task"
                log_dir="${LOG_DIR:-scripts/logs}/$backend/$experiment/$model_suffix/$task"
                mkdir -p "$result_dir" "$score_dir" "$log_dir"
                output="$result_dir/$format.json"
                printf '[%s] %s / %s / %s / %s\n' "$(date -Iseconds)" "$model" "$task" "$format" "$experiment"
                if [[ "$backend" != eval ]]; then
                    common=(--model "$model" --input_json "$input_json" --task "$task" --table_format "$format" --data_dir "$DATA_DIR" --subset "${SUBSET:-auto}" --output "$output")
                    if [[ "$backend" == api ]]; then
                        runner=inference.call_api
                        "$PYTHON" -m "$runner" --experiment "$experiment" "${common[@]}" 2>&1 | tee "$log_dir/${format}_run.log"
                    else
                        runner=inference.run_local_model
                        "$PYTHON" -m "$runner" --experiment "$experiment" "${common[@]}" --model_path "${MODEL_PATH:-$model}" --gpu_num "$gpu_num" --reasoning "$reasoning" 2>&1 | tee "$log_dir/${format}_run.log"
                    fi
                fi
                reference_mode=label
                if [[ "$task" == qa ]]; then reference_mode="${QA_EVAL:-refusal}"; fi
                "$PYTHON" -m evaluation.evaluate_results --reference_mode "$reference_mode" --model "$model" --input_json "$output" --task "$task" --table_format "$format" --reasoning "$reasoning" --output_csv "$score_dir/$format.csv" --output_json "$score_dir/${format}_scores.json" 2>&1 | tee "$log_dir/${format}_eval.log"
            done
        done
    done
done
