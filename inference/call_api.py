import json
import argparse
import os
import datetime
import time
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import openai
from openai import OpenAI
from inference.utils import (file_extensions, read_file_content, count_openai_tokens,
                             append_to_jsonl, convert_jsonl_to_json, save_list_as_jsonl, load_dataset)
from inference.prompts import MAIN_PROMPTS, METHOD_PROMPTS

total_input_tokens = 0
total_output_tokens = 0
processed_count = 0

def call_openai(l_data):
    
    global total_input_tokens, total_output_tokens, processed_count

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    filename = l_data['FileName']
    file_suffix = file_extensions[args.table_format]
    file_path = f"{args.data_dir}/{args.table_format}/{filename}.{file_suffix}"     
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    try:
        content = read_file_content(file_path)
    except Exception as e:
        raise RuntimeError(f"Cannot read table: {file_path}") from e
    prompt = template1.format(table=content, question=l_data["Question"])
    messages = [{"role": "user", "content": prompt}]
    
    input_token_count = count_openai_tokens(prompt, args.model)
    initial_messages = list(messages)
    retries = 0
    while retries < args.max_retries:
        messages = list(initial_messages)

        try:
            
            chat_completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                tools=None,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                top_p=args.top_p,
            )
            response_body = chat_completion.choices[0].message.content
            if hasattr(chat_completion, 'usage') and chat_completion.usage:
                usage = chat_completion.usage
                actual_input_tokens = usage.prompt_tokens
                actual_output_tokens = usage.completion_tokens
            else:
                actual_input_tokens = input_token_count
                actual_output_tokens = count_openai_tokens(response_body, args.model)

            response_body2 = response_body
            if args.experiment == 'method':
                messages.append({"role": "assistant", "content": response_body})
                messages.append({"role": "user", "content": template2})
                chat_completion = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    tools=None,
                    temperature=args.temperature,
                    max_tokens=args.max_new_tokens,
                    top_p=args.top_p,
                )
                response_body2 = chat_completion.choices[0].message.content
                if hasattr(chat_completion, 'usage') and chat_completion.usage:
                    usage = chat_completion.usage
                    actual_input_tokens += usage.prompt_tokens
                    actual_output_tokens += usage.completion_tokens
                else:
                    actual_input_tokens += input_token_count
                    actual_output_tokens += count_openai_tokens(response_body2, args.model)
            total_input_tokens += actual_input_tokens
            total_output_tokens += actual_output_tokens
            append_to_jsonl(
                {
                    "id": l_data["id"],
                    "Table": l_data["FileName"],
                    "ErrorType": l_data["ErrorType"],
                    "ErrorLocation": l_data["ErrorLocation"],
                    "Label": l_data["Label"],  
                    **({"Model_Report": response_body} if args.experiment == "method" else {}),
                    "Model_Answer": response_body2,
                    "input_tokens": actual_input_tokens,
                    "output_tokens": actual_output_tokens,
                    "total_tokens": actual_input_tokens + actual_output_tokens
                },
                output_json,
            )
            processed_count += 1
            p_bar.update(1)
            break

        except Exception as e:

            if isinstance(e, openai.BadRequestError):
                append_to_jsonl(
                    {
                        "id": l_data["id"],
                        "Table": l_data["FileName"],
                        "input": messages,
                        "ErrorType": l_data["ErrorType"],
                        "Model_Answer": "",
                        "Label": l_data["Label"],  
                        "input_tokens": input_token_count,
                        "output_tokens": 0,
                        "total_tokens": input_token_count
                    },
                    output_json,
                )
                p_bar.update(1)
                break

            retries += 1
            print(
                f"Error on call attempt {retries}: {e}"
            )  
            if retries == args.max_retries:
                print(f"Retry exceed the max_retries {retries} times.")
                raise
            time.sleep(args.sleep)

def main(argv=None):
    global args, template1, template2, output_json, p_bar
    global total_input_tokens, total_output_tokens, processed_count

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "method"], default="main")
    parser.add_argument('--input_json', '-i', type=str, default='data/all.json', 
                       help='输入文件')
    parser.add_argument('--model', '-m', type=str, required=True, 
                       help='测试模型')
    parser.add_argument('--output', '-o', type=str, 
                       help='输出路径')
    parser.add_argument('--task', '-t', choices=['qa', 'detect'], required=True, 
                       help='任务')
    parser.add_argument('--table_format', '-f', choices=list(file_extensions), default='csv', 
                       help='表格格式')
    parser.add_argument('--base_url', '-u', type=str, default=os.getenv('OPENAI_BASE_URL'), 
                       help='API网站')
    parser.add_argument('--api_key', '-k', type=str, default=os.getenv('OPENAI_API_KEY'), 
                       help='API Key')

    parser.add_argument("--max_new_tokens", "-max_new_tokens", type=int, default=10240)
    parser.add_argument("--top_p", "-top_p", type=float, default=1)
    parser.add_argument("--temperature", "-temperature", type=float, default=0)
    parser.add_argument("--max_loop", "-max_loop", type=int, default=1)
    parser.add_argument("--max_retries", "-max_retries", type=int, default=1)
    parser.add_argument("--max_threads", "-max_threads", type=int, default=8)
    parser.add_argument("--sleep", "-sleep", type=int, default=2)
    parser.add_argument('--data_dir', default='data', help='Directory containing table format subdirectories')
    parser.add_argument('--subset', choices=['auto', 'all', 'wrong', 'correct'], default='auto',
                        help='auto uses Wrong samples for qa and all samples for detect')
    args = parser.parse_args(argv)

    template1 = METHOD_PROMPTS["first_round"] if args.experiment == "method" else MAIN_PROMPTS[args.task]
    template2 = METHOD_PROMPTS[args.task]

    if args.output:
        output_json=args.output
    else:
        output_json = f"./inference/results/api/{args.experiment}/{args.model.replace(chr(47), chr(95))}/{args.task}/{args.table_format}.json"
    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)

    requests = load_dataset(args.input_json, args.task, args.subset)
    
    total_input_tokens = 0
    total_output_tokens = 0
    processed_count = 0
    if min(args.max_loop, args.max_retries, args.max_threads) < 1:
        parser.error('Loop, retry and thread counts must be positive')
    if not requests:
        if not Path(output_json).exists():
            Path(output_json).write_text('[]\n', encoding='utf-8')
        return
    tstart = datetime.datetime.now()
    for _ in range(args.max_loop):
        try:
            exist_ids = set()
            if os.path.isfile(output_json):
                with open(output_json) as f:
                    for line in f.readlines():
                        l_data = json.loads(line)
                        exist_ids.add(l_data["id"])
        
        except json.decoder.JSONDecodeError:
            existing_data = json.load(open(output_json, "r", encoding="utf-8"))
            exist_ids = {l_data["id"] for l_data in existing_data}
            save_list_as_jsonl(output_json, existing_data)

        dataset = []
        
        for l_data in requests:
            if l_data["id"] not in exist_ids:
                dataset.append(l_data)
        print("EXISTING DATA #: ", len(exist_ids))
        print("API Call REQUEST #: ", len(dataset))
        assert len(exist_ids) + len(dataset) == len(requests)

        if len(dataset) == 0:
            print("All data instances have been processed! Skipping...")
            break

        p_bar = tqdm(range(len(dataset)))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
                list(executor.map(call_openai, dataset))
        finally:
            p_bar.close()

    tend = datetime.datetime.now()
    ttime = tend - tstart
    print("Total time for {} calls: {}".format(len(dataset), ttime))
    
    convert_jsonl_to_json(output_json)

    print(f"\n=== Token使用统计 ===")
    print(f"本次处理数据量: {processed_count}")
    print(f"累计总数据量: {len(requests)}")
    if processed_count > 0:
        print(f"输入tokens总计: {total_input_tokens}")
        print(f"输出tokens总计: {total_output_tokens}")
        print(f"总tokens: {total_input_tokens + total_output_tokens}")
        print(f"平均输入tokens/样本: {total_input_tokens/processed_count:.2f}")
        print(f"平均输出tokens/样本: {total_output_tokens/processed_count:.2f}")
        print(f"平均总tokens/样本: {(total_input_tokens + total_output_tokens)/processed_count:.2f}")


if __name__ == "__main__":
    main()
