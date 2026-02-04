#!/usr/bin/env python3
"""
Estimate theoretical R² ceiling using k-NN neighbor variance analysis.

Usage:
    python ceiling_analysis.py --data data.csv --target True_HOMA_IR --k 10

For each sample, finds k nearest neighbors in standardized feature space
and computes target variance among neighbors. This neighbor variance is
the irreducible noise — no model can resolve target differences among
feature-space neighbors.

    R²_max = 1 - mean(neighbor_variance) / var(y_total)
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def estimate_ceiling(X, y, k=10):
    """Estimate theoretical R² ceiling via k-NN neighbor variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    # Exclude self (index 0)
    neighbor_indices = indices[:, 1:]

    neighbor_variances = []
    for i in range(len(y)):
        neighbor_targets = y[neighbor_indices[i]]
        neighbor_variances.append(np.var(neighbor_targets))

    noise_var = np.mean(neighbor_variances)
    total_var = np.var(y)
    r2_max = 1 - noise_var / total_var

    return {
        'r2_max': r2_max,
        'noise_var': noise_var,
        'total_var': total_var,
        'noise_ratio': noise_var / total_var,
        'k': k,
        'n_samples': len(y),
        'n_features': X.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description='Estimate R² ceiling')
    parser.add_argument('--data', required=True, help='Path to CSV')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--drop', nargs='*', default=[], help='Columns to drop')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.target].values
    drop_cols = [args.target] + args.drop
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categoricals
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes

    X = X.fillna(X.median())

    result = estimate_ceiling(X.values, y, k=args.k)

    print(f"\n{'='*50}")
    print(f"Theoretical Ceiling Analysis (k={result['k']})")
    print(f"{'='*50}")
    print(f"Samples: {result['n_samples']}, Features: {result['n_features']}")
    print(f"Target variance: {result['total_var']:.4f}")
    print(f"Neighbor noise variance: {result['noise_var']:.4f}")
    print(f"Noise ratio: {result['noise_ratio']:.3f}")
    print(f"Theoretical max R²: {result['r2_max']:.4f}")
    print(f"{'='*50}\n")

    # Sensitivity analysis
    print("Sensitivity to k:")
    for k_test in [5, 10, 15, 20, 30]:
        r = estimate_ceiling(X.values, y, k=k_test)
        print(f"  k={k_test:2d}: R²_max = {r['r2_max']:.4f}")


if __name__ == '__main__':
    main()
