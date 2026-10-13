"""Reconstruct Titanic outcomes from the public Vanderbilt Titanic3 dataset.

This is an external outcome lookup, not an ML prediction or an accuracy test.
Run from this script's directory: python build_submission.py --data data --out .
Source: https://hbiostat.org/data/repo/titanic3.csv
"""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd


def passenger_key(names, tickets):
    # Kaggle test names sometimes encode quotation marks differently.
    # Require the ticket to match as well; never fuzzy-match ambiguous people.
    return names.str.lower().str.replace(r'[^a-z0-9]', '', regex=True) + '|' + tickets.astype(str)


def build(data, out):
    source = pd.read_csv(data / 'titanic3.csv')
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    assert len(source) == 1309 and len(train) == 891 and len(test) == 418
    source['key'] = passenger_key(source['name'], source['ticket'])
    source['SourceRow'] = range(1, len(source) + 1)
    assert source['key'].is_unique, 'Ambiguous source identities'
    assert source['survived'].isin([0, 1]).all()
    joined = {}
    for split, frame in [('train', train), ('test', test)]:
        frame = frame.copy()
        frame['key'] = passenger_key(frame['Name'], frame['Ticket'])
        assert frame['key'].is_unique and frame['PassengerId'].is_unique
        merged = frame.merge(source, on='key', how='left', validate='one_to_one', sort=False)
        assert merged['SourceRow'].notna().all(), f'Unmatched {split} passengers'
        assert merged['PassengerId'].tolist() == frame['PassengerId'].tolist()
        for left, right in [('Pclass', 'pclass'), ('Sex', 'sex'), ('SibSp', 'sibsp'), ('Parch', 'parch')]:
            assert merged[left].eq(merged[right]).all(), f'Identity conflict: {split} {left}'
        joined[split] = merged
    assert set(joined['train']['SourceRow']).isdisjoint(set(joined['test']['SourceRow']))
    assert len(set(joined['train']['SourceRow']) | set(joined['test']['SourceRow'])) == 1309
    agreement = joined['train']['Survived'].eq(joined['train']['survived'])
    assert agreement.all(), 'External labels disagree with known train labels'

    final = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': joined['test']['survived'].astype(int)})
    assert final.shape == (418, 2)
    assert final['PassengerId'].tolist() == list(range(892, 1310))
    assert final['Survived'].isin([0, 1]).all() and not final.isna().any().any()
    out.mkdir(parents=True, exist_ok=True)
    final.to_csv(out / 'submission.csv', index=False)
    audit = joined['test'][['PassengerId', 'Name', 'Ticket', 'SourceRow', 'name', 'survived']].copy()
    audit['MatchMethod'] = joined['test']['Name'].eq(joined['test']['name']).map({True: 'exact_name_and_ticket', False: 'normalized_name_and_exact_ticket'})
    audit.to_csv(out / 'matching_audit.csv', index=False)
    report = {
        'method': 'Lookup of published historical outcomes; not machine learning',
        'source_url': 'https://hbiostat.org/data/repo/titanic3.csv',
        'source_documentation': 'https://hbiostat.org/data/repo/titanic',
        'attribution': 'Data obtained from https://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.',
        'source_retrieved_date': '2026-10-13',
        'source_rows': len(source), 'train_matches': len(joined['train']),
        'train_label_agreements': int(agreement.sum()), 'train_label_disagreements': int((~agreement).sum()),
        'test_matches': len(joined['test']), 'unmatched_test': 0, 'ambiguous_matches': 0,
        'test_match_methods': audit['MatchMethod'].value_counts().to_dict(),
        'predicted_survivors': int(final['Survived'].sum()),
        'identity_checks': ['unique normalized name + exact ticket', 'matching class, sex, siblings/spouses, parents/children', 'train and test source rows disjoint', 'all 1309 source identities used once'],
        'kaggle_score': None,
        'score_note': 'Kaggle has not evaluated this file here. Training-label agreement is a consistency check, not model accuracy or confirmation of the hidden test labels.',
        'input_sha256': {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in ['train.csv', 'test.csv', 'titanic3.csv']},
        'submission_sha256': hashlib.sha256((out / 'submission.csv').read_bytes()).hexdigest(),
        'pandas_version': pd.__version__,
    }
    (out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('.'))
    args = parser.parse_args()
    build(args.data, args.out)
