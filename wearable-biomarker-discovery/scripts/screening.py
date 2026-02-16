#!/usr/bin/env python3
"""Wearable biomarker screening pipeline.

Usage: python screening.py <data_csv> <outcome_col> [--threshold 10] [--output-dir output/]

Runs Spearman correlations, FDR correction, OR per SD, and redundancy analysis.
Outputs: screening_results.csv, independent_features.json, consort_flow.txt
"""
import argparse, json, sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Wearable biomarker screening')
    parser.add_argument('data', help='CSV file with subjects as rows')
    parser.add_argument('outcome', help='Outcome column name')
    parser.add_argument('--threshold', type=float, default=10, help='Binary threshold (default: 10)')
    parser.add_argument('--feature-prefix', nargs='+', default=['rhr_','sleep_','steps_','hrv_','cardio_','respiratory_'],
                        help='Feature column prefixes')
    parser.add_argument('--output-dir', default='output/', help='Output directory')
    parser.add_argument('--missing-threshold', type=float, default=0.5, help='Max missing rate per subject')
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.data)
    n_raw = len(df)
    print(f"Raw: N={n_raw}, columns={len(df.columns)}")

    # Identify features
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in args.feature_prefix)]
    print(f"Features: {len(feat_cols)}")

    # Clean
    df = df.dropna(subset=[args.outcome])
    n_outcome = len(df)
    missing_rate = df[feat_cols].isnull().mean(axis=1)
    df = df[missing_rate < args.missing_threshold]
    n_clean = len(df)

    binary = (df[args.outcome] >= args.threshold).astype(int)
    n_cases = binary.sum()
    prev = binary.mean()
    print(f"Clean: N={n_clean}, cases={n_cases} ({prev:.1%})")

    # CONSORT flow
    with open(outdir / 'consort_flow.txt', 'w') as f:
        f.write(f"Raw: {n_raw}\n")
        f.write(f"After outcome filter: {n_outcome}\n")
        f.write(f"After missing filter (<{args.missing_threshold}): {n_clean}\n")
        f.write(f"Cases (>={args.threshold}): {n_cases} ({prev:.1%})\n")

    # Screening
    results = []
    scaler = StandardScaler()
    for feat in feat_cols:
        valid = df[feat].notna()
        if valid.sum() < 100:
            continue
        rho, p = spearmanr(df.loc[valid, feat], binary[valid])
        # OR per SD
        X = scaler.fit_transform(df.loc[valid, [feat]])
        y = binary[valid].values
        try:
            lr = LogisticRegression(max_iter=1000).fit(X, y)
            or_sd = np.exp(lr.coef_[0][0])
        except:
            or_sd = np.nan
        results.append({'feature': feat, 'rho': rho, 'p': p, 'n': int(valid.sum()), 'OR_per_SD': or_sd})

    res = pd.DataFrame(results)
    _, res['p_fdr'], _, _ = multipletests(res['p'], method='fdr_bh')
    res = res.sort_values('p_fdr')
    res.to_csv(outdir / 'screening_results.csv', index=False)

    sig = res[res['p_fdr'] < 0.05]
    print(f"FDR-significant: {len(sig)}/{len(res)}")

    # Redundancy — correlation among top 30 features
    top = sig.head(30)['feature'].tolist()
    if len(top) > 1:
        corr = df[top].corr(method='spearman').abs()
        # Greedy pruning: drop features with r > 0.85 to any already-kept feature
        kept = [top[0]]
        for feat in top[1:]:
            if all(corr.loc[feat, k] < 0.85 for k in kept):
                kept.append(feat)
        print(f"Independent features after pruning: {len(kept)}")
        json.dump(kept, open(outdir / 'independent_features.json', 'w'))
    else:
        kept = top
        json.dump(kept, open(outdir / 'independent_features.json', 'w'))

    # Save clean data
    df.to_csv(outdir / 'subjects_clean.csv', index=False)
    json.dump(feat_cols, open(outdir / 'feat_cols.json', 'w'))
    print(f"\nOutputs saved to {outdir}/")

if __name__ == '__main__':
    main()
