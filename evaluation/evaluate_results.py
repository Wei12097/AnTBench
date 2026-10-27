"""Evaluate QA or detection with explicit reference-answer semantics."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from evaluation.metrics.qa_metrics import QAMetric
from inference.utils import get_final_answer, normalize_label


def fixed_reference_answer(response):
    """Preserve the original fixed-reference evaluator's answer extraction."""
    answer = ''
    if response:
        if '[Final Answer]:' in response:
            answer = response.split('[Final Answer]:')[-1].strip()
        elif 'Final Answer:' in response:
            answer = response.split('Final Answer:')[-1].strip()
        answer = answer.replace('"', '')
        if '\n' in answer:
            answer = answer.split('\n')[0].strip()
    return answer


def classification_scores(frame):
    valid = frame[frame['Normalized_Label'].isin(['Correct', 'Wrong'])]
    if valid.empty:
        return {}
    true, pred = valid['Normalized_Label'], valid['Binary_Prediction']
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, pred, labels=['Wrong', 'Correct'], average=None, zero_division=0)
    _, _, macro, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)
    groups = {}
    for error_type, group in valid.groupby('ErrorType'):
        pairs = list(zip(group['Normalized_Label'], group['Binary_Prediction']))
        tp = sum(t == 'Wrong' and p == 'Wrong' for t, p in pairs)
        fn = sum(t == 'Wrong' and p == 'Correct' for t, p in pairs)
        fp = sum(t == 'Correct' and p == 'Wrong' for t, p in pairs)
        sub_precision = tp / (tp + fp) * 100 if tp + fp else 0.0
        sub_recall = tp / (tp + fn) * 100 if tp + fn else 0.0
        groups[str(error_type)] = (2 * sub_precision * sub_recall / (sub_precision + sub_recall)
                                   if sub_precision + sub_recall else 0.0)
    return {'wrong_precision': float(precision[0] * 100),
            'wrong_recall': float(recall[0] * 100), 'wrong_f1': float(f1[0] * 100),
            'correct_f1': float(f1[1] * 100), 'macro_f1': float(macro * 100),
            'type_specific_f1': groups}


def evaluate_records(data, reference_mode):
    if not data:
        raise ValueError('Cannot evaluate an empty result set')
    metric = QAMetric()
    rows = []
    for item in data:
        reference = item['Label'] if reference_mode == 'label' else (
            'Refuse to answer' if reference_mode == 'refusal' else 'Wrong')
        prediction = (get_final_answer if reference_mode == 'label' else fixed_reference_answer)(item['Model_Answer'])
        scores = metric.compute([reference], [prediction])
        row = {'Id': item['id'], 'Table': item['Table'],
               'ErrorType': item.get('ErrorType', 'None'), 'Label': reference,
               'Model_Answer': prediction, 'EM': scores['EM']}
        if reference_mode == 'label':
            label = normalize_label(reference)
            normalized = normalize_label(prediction)
            binary = normalized
            if label in ['Correct', 'Wrong'] and binary not in ['Correct', 'Wrong']:
                binary = 'Correct' if label == 'Wrong' else 'Wrong'
            row.update(Normalized_Label=label, Normalized_Prediction=normalized, Binary_Prediction=binary)
        else:
            row['F1'] = scores['F1']
        rows.append(row)
    frame = pd.DataFrame(rows)
    grouped = frame.copy()
    grouped['ErrorType'] = grouped['ErrorType'].replace('Statistics', 'Calculation')
    if reference_mode == 'label':
        summary = {'mean': float(frame['EM'].mean()), **classification_scores(grouped)}
    else:
        summary = {'overall_f1': float(frame['F1'].mean()), 'overall_em': float(frame['EM'].mean()),
                   'error_type_f1': grouped.groupby('ErrorType')['F1'].mean().to_dict(),
                   'error_type_em': grouped.groupby('ErrorType')['EM'].mean().to_dict()}
    return frame, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_json', '-i', required=True)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--task', '-t', choices=['qa', 'detect'], required=True)
    parser.add_argument('--table_format', '-f', choices=['csv', 'html', 'markdown', 'latex'], default='csv')
    parser.add_argument('--reasoning', '-r', type=str.lower, choices=['true', 'false'], default='false')
    parser.add_argument('--reference_mode', choices=['auto', 'label', 'refusal', 'wrong'], default='auto',
                        help='auto: QA refusal reference; detection per-record Label')
    parser.add_argument('--output_csv')
    parser.add_argument('--output_json', default='evaluation/results/scores.json')
    args = parser.parse_args(argv)
    mode = args.reference_mode
    if mode == 'auto':
        mode = 'refusal' if args.task == 'qa' else 'label'
    model_suffix = args.model + ('_reasoning' if args.reasoning == 'true' else '')
    data = json.loads(Path(args.input_json).read_text(encoding='utf-8'))
    frame, summary = evaluate_records(data, mode)
    output = Path(args.output_csv or f'evaluation/results/{model_suffix.replace(chr(47), chr(95))}/{args.task}/{args.table_format}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary_path = Path(args.output_json)
    summaries = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    summaries.setdefault(model_suffix, {}).setdefault(args.task, {})[args.table_format] = summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=4), encoding='utf-8')
    print(f'Reference mode: {mode}\n{json.dumps(summary, ensure_ascii=False, indent=2)}')


if __name__ == '__main__':
    main()
