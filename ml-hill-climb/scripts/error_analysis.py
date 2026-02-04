#!/usr/bin/env python3
"""
Comprehensive error analysis for regression models.

Usage:
    python error_analysis.py --data data.csv --target y --predictions preds.npz

Produces:
1. Residual-feature correlations (is there unused signal?)
2. Subgroup performance breakdown
3. Error correlation between models (ensemble diversity check)
4. Prediction range analysis
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def residual_feature_correlations(X, residuals, feature_names=None):
    """Check if residuals correlate with any feature (unused signal)."""
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(X.shape[1])]

    correlations = []
    for i in range(X.shape[1]):
        r, p = pearsonr(X[:, i], residuals)
        correlations.append({'feature': feature_names[i], 'correlation': r, 'p_value': p})

    df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
    return df


def subgroup_analysis(y_true, y_pred, groups, group_name='group'):
    """Performance breakdown by subgroup."""
    results = []
    for g in sorted(groups.unique()):
        mask = groups == g
        n = mask.sum()
        if n < 10:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        mae = np.mean(np.abs(yt - yp))
        bias = np.mean(yt - yp)
        results.append({group_name: g, 'n': n, 'R2': r2, 'MAE': mae, 'Bias': bias})
    return pd.DataFrame(results)


def error_correlation_matrix(predictions_dict, y_true):
    """Compute pairwise error correlations between models."""
    model_names = list(predictions_dict.keys())
    errors = {name: y_true - pred for name, pred in predictions_dict.items()}

    n = len(model_names)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = np.corrcoef(errors[model_names[i]], errors[model_names[j]])[0, 1]

    return pd.DataFrame(corr_matrix, index=model_names, columns=model_names)


def homa_range_analysis(y_true, y_pred):
    """Error breakdown by target range bins."""
    bins = [0, 1, 2, 3, 5, 8, float('inf')]
    labels = ['[0,1)', '[1,2)', '[2,3)', '[3,5)', '[5,8)', '[8+)']
    residuals = y_true - y_pred

    results = []
    for i in range(len(bins) - 1):
        mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        results.append({
            'range': labels[i], 'n': n,
            'MAE': np.mean(np.abs(residuals[mask])),
            'Bias': np.mean(residuals[mask]),
            'RMSE': np.sqrt(np.mean(residuals[mask] ** 2)),
        })
    return pd.DataFrame(results)


def full_analysis(y_true, y_pred, X=None, feature_names=None):
    """Run complete error analysis and print summary."""
    residuals = y_true - y_pred
    r2 = 1 - np.sum(residuals ** 2) / np.sum((y_true - y_true.mean()) ** 2)

    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS REPORT")
    print(f"{'=' * 60}")
    print(f"R²: {r2:.4f}  |  MAE: {np.mean(np.abs(residuals)):.4f}  |  RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"Pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}] vs True: [{y_true.min():.2f}, {y_true.max():.2f}]")
    print(f"Corr(residual, y_true): {np.corrcoef(y_true, residuals)[0,1]:.3f}")

    if X is not None:
        print(f"\n--- Residual-Feature Correlations (top 10) ---")
        rfc = residual_feature_correlations(X, residuals, feature_names)
        print(rfc.head(10).to_string(index=False))
        max_corr = rfc['correlation'].abs().max()
        if max_corr < 0.05:
            print("✓ All correlations near zero — signal appears exhausted")
        else:
            print(f"⚠ Max |correlation| = {max_corr:.3f} — possible unused signal")

    print(f"\n--- Target Range Breakdown ---")
    print(homa_range_analysis(y_true, y_pred).to_string(index=False))
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--y-true', required=True, help='Path to true values (.npy)')
    parser.add_argument('--y-pred', required=True, help='Path to predictions (.npy)')
    args = parser.parse_args()

    y_true = np.load(args.y_true)
    y_pred = np.load(args.y_pred)
    full_analysis(y_true, y_pred)
