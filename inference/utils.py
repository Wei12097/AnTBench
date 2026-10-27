import pandas as pd
from pathlib import Path
import re
import json
import tiktoken
import os

file_extensions = {
    "latex": "tex",
    "markdown": "md",
    "csv": "csv",
    "html": "html"
}

def read_file_content(file_path):
    """
    根据文件扩展名读取不同格式的文件
    
    参数:
        file_path: 文件路径
    
    返回:
        文件内容的字符串表示
    """
    file_ext = Path(file_path).suffix.lower()
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
        return df.rename(columns=lambda x: '' if 'Unnamed' in x else x).to_csv(index=False)
    
    elif file_ext in ['.html', '.htm', '.tex', '.latex', '.md', '.markdown', '.txt', '.json']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        return df.rename(columns=lambda x: '' if 'Unnamed' in x else x).to_csv(index=False)
    
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"不支持的文件格式: {file_ext}, 错误: {str(e)}")

def normalize_label(text):
    """
    规范化标签，提取Correct或Wrong
    处理各种格式如: [Wrong], **Correct**, Wrong., etc.
    """
    if not text:
        return "Unknown"

    text_lower = str(text).lower()

    text_clean = re.sub(r'[\[\]\*\.\,\!\?\:\;\"\']+', '', text_lower).strip()

    if 'wrong' in text_clean:
        return "Wrong"
    
    elif 'correct' in text_clean:
        return "Correct"
    else:
        
        if 'error' in text_clean or 'incorrect'in text_clean:
            return "Wrong"
        elif 'right' in text_clean:
            return "Correct"
        else:
            return "Unknown"

def get_final_answer(response):
    """提取最终答案"""
    final_answer = ""
    if response:
        if "[Final Answer]:" in response:
            final_answer = response.split("[Final Answer]:")[-1].strip()
        elif "**Final Answer**:" in response:
            final_answer = response.split("**Final Answer**:")[-1].strip()
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        elif "Final Answer" in response:
            final_answer = response.split("Final Answer")[-1].strip()
        final_answer = final_answer.replace("\"", "")
        if "\n" in final_answer:
            final_answer = final_answer.split("\n")[0].strip()
    return final_answer

def count_openai_tokens(text, model):
    """使用tiktoken统计OpenAI模型的token数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def convert_jsonl_to_json(file_path) -> None:
    """Convert a jsonl file to a json file."""
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

def save_list_as_jsonl(path: str, data):
    with open(path, "w", encoding="utf8") as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write("\n")
    print(f"Saved {len(data)} data to {path}")


def load_dataset(path, task, subset='auto'):
    """Select anomaly-only perception data or balanced detection data."""
    with open(path, encoding='utf-8') as handle:
        records = json.load(handle)
    ids = [item['id'] for item in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Dataset sample IDs must be unique')
    if subset == 'auto':
        subset = 'wrong' if task == 'qa' else 'all'
    if subset != 'all':
        label = 'Wrong' if subset == 'wrong' else 'Correct'
        records = [item for item in records if item['Label'] == label]
    return records
