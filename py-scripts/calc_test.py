import pretty_errors
import click
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from pathlib import Path

@click.command()
@click.option('--test_jsonl', type=str, help='Path to test_seen.jsonl or test_unseen.jsonl')
@click.option('--result_csv', type=str, help='Path to result csv of model that is tested on seen or unseen')
def calc_test(test_jsonl, result_csv):
    phase_cases = {'test_seen.jsonl':1, 'test_unseen.jsonl': 2}
    try:
        case = phase_cases[Path(test_jsonl).name]
    except KeyError:
        case = '_'

    test_df = pd.read_json(test_jsonl, lines=True)
    result_df = pd.read_csv(result_csv)

    if not tuple(test_df['id'].tolist()) == tuple(result_df['id'].tolist()):
        result_df = result_df.set_index('id')
        result_df = result_df.reindex(index=test_df['id'])
        result_df = result_df.reset_index()
    
    roc_auc = roc_auc_score(test_df['label'], result_df['proba'])
    accuracy = accuracy_score(test_df['label'], result_df['label'])
    print(f'Phase: {case}; AUC ROC: {roc_auc:.4f}; Accuracy: {accuracy:.4f}')

if '__main__' == __name__:
    calc_test()