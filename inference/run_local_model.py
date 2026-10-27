"""Run the main experiment or two-round method with a local chat model."""
import argparse
import json
from pathlib import Path

from inference.prompts import MAIN_PROMPTS, METHOD_PROMPTS
from inference.utils import file_extensions, read_file_content, load_dataset


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['main', 'method'], default='main')
    parser.add_argument('--input_json', '-i', default='data/all.json')
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--model_path', help='Local weights or repository ID; defaults to --model')
    parser.add_argument('--output', '-o')
    parser.add_argument('--task', '-t', choices=['qa', 'detect'], required=True)
    parser.add_argument('--table_format', '-f', choices=list(file_extensions), default='csv')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--gpu_num', '-n', type=int, default=1)
    parser.add_argument('--reasoning', '-r', type=str.lower, choices=['true', 'false'], default='false')
    parser.add_argument('--subset', choices=['auto', 'all', 'wrong', 'correct'], default='auto')
    args = parser.parse_args(argv)
    if args.gpu_num < 1:
        parser.error('--gpu_num must be positive')
    thinking = args.reasoning == 'true'
    model_suffix = args.model.replace('/', '_') + ('_reasoning' if thinking else '')
    output_path = Path(args.output or f'inference/results/local/{args.experiment}/{model_suffix}/{args.task}/{args.table_format}.json')
    dataset = load_dataset(args.input_json, args.task, args.subset)
    existing = json.loads(output_path.read_text(encoding='utf-8')) if output_path.exists() else []
    processed_ids = {item['id'] for item in existing}
    pending = [item for item in dataset if item['id'] not in processed_ids]
    print(f'Pending samples: {len(pending)}')
    if not pending:
        if not output_path.exists():
            save_json(output_path, [])
        return

    prompt = MAIN_PROMPTS[args.task] if args.experiment == 'main' else METHOD_PROMPTS['first_round']
    histories = []
    for item in pending:
        table_path = Path(args.data_dir) / args.table_format / f"{item['FileName']}.{file_extensions[args.table_format]}"
        content = read_file_content(table_path)
        histories.append([{'role': 'user', 'content': prompt.format(table=content, question=item['Question'])}])

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path or args.model, trust_remote_code=True,
              max_model_len=32768, tensor_parallel_size=args.gpu_num,
              max_num_seqs=2 * args.gpu_num,
              gpu_memory_utilization=0.68 if args.experiment == 'main' else 0.7,
              dtype='auto')
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0, top_p=1, max_tokens=10240, seed=24)

    def generate(messages):
        prompts = [tokenizer.apply_chat_template(history, tokenize=False,
                   add_generation_prompt=True, enable_thinking=thinking, tools=None)
                   for history in messages]
        outputs = llm.generate(prompts, sampling)
        if len(outputs) != len(messages):
            raise RuntimeError('Model returned an unexpected number of responses')
        return outputs

    reports = None
    if args.experiment == 'method':
        cache_path = Path(str(output_path) + '.round1.json')
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding='utf-8'))
                if ([row['id'] for row in cached] == [row['id'] for row in pending]
                        and all(isinstance(row['round1_response'], str) for row in cached)):
                    reports = cached
            except (ValueError, KeyError, TypeError):
                pass
        if reports is None:
            first_outputs = generate(histories)
            reports = [{'id': item['id'], 'round1_response': output.outputs[0].text,
                        'prompt_token_ids_count': len(output.prompt_token_ids),
                        'output_token_ids_count': len(output.outputs[0].token_ids)}
                       for item, output in zip(pending, first_outputs)]
            save_json(cache_path, reports)
        for history, report in zip(histories, reports):
            history.extend([{'role': 'assistant', 'content': report['round1_response']},
                            {'role': 'user', 'content': METHOD_PROMPTS[args.task]}])

    outputs = generate(histories)
    new_results = []
    input_tokens = output_tokens = 0
    for index, (item, output) in enumerate(zip(pending, outputs)):
        result = {'id': item['id'], 'Table': item['FileName'],
                  'ErrorType': item['ErrorType'], 'ErrorLocation': item['ErrorLocation'],
                  'Label': item['Label'], 'Model_Answer': output.outputs[0].text}
        input_tokens += len(output.prompt_token_ids)
        output_tokens += len(output.outputs[0].token_ids)
        if reports is not None:
            result['Model_Report'] = reports[index]['round1_response']
            input_tokens += reports[index].get('prompt_token_ids_count', 0)
            output_tokens += reports[index].get('output_token_ids_count', 0)
        new_results.append(result)
    save_json(output_path, existing + new_results)
    print(f'Saved {len(new_results)} new samples to {output_path}')
    print(f'Input tokens: {input_tokens}; output tokens: {output_tokens}')


if __name__ == '__main__':
    main()
