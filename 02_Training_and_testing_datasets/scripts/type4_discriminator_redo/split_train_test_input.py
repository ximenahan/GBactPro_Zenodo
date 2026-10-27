#!/usr/bin/env python3
"""Split species TSS tables into train/test (75/25) before dataset generation."""

from pathlib import Path
import pandas as pd

INPUT_DIR = Path('../input/scan_result_100')
TRAIN_DIR = Path('../input_split/train/scan_result_100')
TEST_DIR = Path('../input_split/test/scan_result_100')
SUMMARY = Path('../input_split/split_summary.tsv')
SEED = 42
TRAIN_FRAC = 0.75


def main() -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    files = sorted(INPUT_DIR.glob('*.tsv'))
    for fp in files:
        df = pd.read_csv(fp, sep='	')
        n = len(df)
        n_train = int(n * TRAIN_FRAC)

        shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        train_df = shuffled.iloc[:n_train].copy()
        test_df = shuffled.iloc[n_train:].copy()

        train_df.to_csv(TRAIN_DIR / fp.name, sep='	', index=False)
        test_df.to_csv(TEST_DIR / fp.name, sep='	', index=False)

        rows.append({
            'species': fp.stem,
            'total_rows': n,
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'train_fraction': round(len(train_df) / n, 4) if n else 0.0,
        })

    pd.DataFrame(rows).to_csv(SUMMARY, sep='	', index=False)
    print(f'Split complete: {len(files)} species')
    print(f'Train files: {TRAIN_DIR}')
    print(f'Test files:  {TEST_DIR}')
    print(f'Summary:     {SUMMARY}')


if __name__ == '__main__':
    main()
