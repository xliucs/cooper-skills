#!/usr/bin/env python3
"""Validation split and feature replication tracking.

Usage: python validate.py <clean_csv> <outcome_col> <features_json> [--threshold 10] [--test-size 0.4]

Outputs: validation_results.csv, replication_summary.txt
"""
import argparse, json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Clean CSV')
    parser.add_argument('outcome', help='Outcome column')
    parser.add_argument('features', help='JSON file with feature list')
    parser.add_argument('--threshold', type=float, default=10)
    parser.add_argument('--test-size', type=float, default=0.4)
    parser.add_argument('--output-dir', default='output/')
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    feats = json.load(open(args.features))
    binary = (df[args.outcome] >= args.threshold).astype(int)

    # Drop rows with any NaN in selected features
    mask = df[feats].notna().all(axis=1)
    df_clean = df[mask]
    y = binary[mask]

    X_disc, X_val, y_disc, y_val = train_test_split(
        df_clean[feats], y, test_size=args.test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_disc_s = scaler.fit_transform(X_disc)
    X_val_s = scaler.transform(X_val)

    # Model
    lr = LogisticRegression(max_iter=1000).fit(X_disc_s, y_disc)
    auc_disc = roc_auc_score(y_disc, lr.predict_proba(X_disc_s)[:, 1])
    auc_val = roc_auc_score(y_val, lr.predict_proba(X_val_s)[:, 1])

    # PPV at various thresholds
    probs = lr.predict_proba(X_val_s)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, probs)

    # Feature replication
    results = []
    replicated = 0
    for feat in feats:
        rho_d, p_d = spearmanr(X_disc[feat], y_disc)
        rho_v, p_v = spearmanr(X_val[feat], y_val)
        rep = (p_v < 0.05) and (np.sign(rho_d) == np.sign(rho_v))
        if rep:
            replicated += 1
        results.append({
            'feature': feat,
            'rho_disc': rho_d, 'p_disc': p_d,
            'rho_val': rho_v, 'p_val': p_v,
            'replicated': rep
        })

    pd.DataFrame(results).to_csv(outdir / 'validation_results.csv', index=False)

    summary = (
        f"Discovery AUC: {auc_disc:.3f}\n"
        f"Validation AUC: {auc_val:.3f}\n"
        f"AUC drop: {auc_disc - auc_val:.3f}\n"
        f"Replicated: {replicated}/{len(feats)} ({replicated/len(feats):.0%})\n"
        f"Prevalence (val): {y_val.mean():.1%}\n"
        f"Max PPV at 50% recall: {prec[rec >= 0.5].max():.3f}\n"
    )
    print(summary)
    with open(outdir / 'replication_summary.txt', 'w') as f:
        f.write(summary)

if __name__ == '__main__':
    main()
