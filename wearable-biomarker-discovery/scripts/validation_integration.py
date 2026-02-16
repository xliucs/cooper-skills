"""
ROUNDS 6-10: Dose-response, autoencoder, cross-system, validation, integration
"""
import pandas as pd, numpy as np, json, time, os
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
import warnings; warnings.filterwarnings('ignore')

OUT = 'output/v5'
subj = pd.read_csv(f'{OUT}/subjects_clean.csv')
feat_cols = json.load(open(f'{OUT}/feat_cols.json'))
feat_cols = [c for c in feat_cols if c in subj.columns]

print(f"Subjects: {len(subj)}, Features: {len(feat_cols)}")

# ============================================================
# ROUND 6: DOSE-RESPONSE WITH RESTRICTED CUBIC SPLINES
# ============================================================
print("\n" + "="*70)
print("ROUND 6: DOSE-RESPONSE ANALYSIS")
print("="*70)

key_biomarkers = ['rhr_bpm_mean','sleep_time_minutes_cv','num_steps_mean',
                  'rmssd_mean','cardio_minutes_cv','peak_intensity_ratio']
key_biomarkers = [b for b in key_biomarkers if b in subj.columns]

dose_results = []
for bm in key_biomarkers:
    valid = subj[[bm, 'phq_score', 'phq10']].dropna()
    if len(valid) < 500: continue
    
    # Quintile analysis
    valid['quintile'] = pd.qcut(valid[bm], 5, labels=False, duplicates='drop')
    
    q_stats = []
    for q in sorted(valid['quintile'].unique()):
        grp = valid[valid['quintile'] == q]
        phq_mean = grp['phq_score'].mean()
        phq10_rate = grp['phq10'].mean()
        bm_mean = grp[bm].mean()
        q_stats.append({'quintile': q+1, 'bm_mean': round(bm_mean, 2),
                        'phq_mean': round(phq_mean, 2), 'phq10_rate': round(phq10_rate, 4)})
    
    # Test for non-linearity: compare Q5 vs Q4 and Q2 vs Q1
    q_df = pd.DataFrame(q_stats)
    q1_rate = q_df.iloc[0]['phq10_rate']
    q5_rate = q_df.iloc[-1]['phq10_rate']
    
    # Polynomial regression for non-linearity test
    from numpy.polynomial import polynomial as P
    x = valid[bm].values
    y = valid['phq_score'].values
    # Linear
    c1 = np.polyfit(x, y, 1)
    y_lin = np.polyval(c1, x)
    ss_lin = np.sum((y - y_lin)**2)
    # Quadratic
    c2 = np.polyfit(x, y, 2)
    y_quad = np.polyval(c2, x)
    ss_quad = np.sum((y - y_quad)**2)
    # F-test for non-linearity
    n = len(valid)
    f_stat = ((ss_lin - ss_quad) / 1) / (ss_quad / (n - 3))
    from scipy.stats import f as f_dist
    p_nonlin = 1 - f_dist.cdf(f_stat, 1, n-3)
    
    dose_results.append({
        'biomarker': bm, 'q1_phq10': q1_rate, 'q5_phq10': q5_rate,
        'risk_ratio_q5_q1': round(q5_rate/max(q1_rate,0.001), 2),
        'nonlinearity_p': round(p_nonlin, 4),
        'quintiles': q_stats
    })
    print(f"  {bm}:")
    print(f"    Q1 dep rate: {q1_rate:.4f}, Q5: {q5_rate:.4f}, RR={q5_rate/max(q1_rate,0.001):.2f}")
    print(f"    Non-linearity F-test p={p_nonlin:.4f} ({'NON-LINEAR' if p_nonlin<0.05 else 'linear'})")

pd.DataFrame([{k:v for k,v in d.items() if k!='quintiles'} for d in dose_results]).to_csv(f'{OUT}/dose_response.csv', index=False)

# ============================================================
# ROUND 7: AUTOENCODER LATENT BIOMARKERS
# ============================================================
print("\n" + "="*70)
print("ROUND 7: AUTOENCODER LATENT BIOMARKERS")
print("="*70)

# Use all wearable mean + CV features
ae_cols = [c for c in feat_cols if (c.endswith('_mean') or c.endswith('_cv') or c.endswith('_std'))
           and not any(x in c for x in ['7day','subj']) and subj[c].notna().sum() > len(subj)*0.8]
valid_ae = subj[ae_cols + ['phq_score','phq10','gad10']].dropna()
X_ae = StandardScaler().fit_transform(valid_ae[ae_cols])
print(f"  AE features: {len(ae_cols)}, subjects: {len(valid_ae)}")

# Bottleneck autoencoder: input -> 32 -> 8 -> 32 -> input
from sklearn.neural_network import MLPRegressor

# Train autoencoder (encode to 8-dim latent)
# We'll use a 2-step approach: first reduce to 8 dims via MLP
# Actually, let's use PCA for comparison and a proper bottleneck
from sklearn.decomposition import PCA

# PCA baseline
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_ae)
print(f"  PCA variance explained (8 components): {pca.explained_variance_ratio_.sum():.3f}")

# MLP autoencoder
ae = MLPRegressor(hidden_layer_sizes=(32, 8, 32), max_iter=500, random_state=42,
                  early_stopping=True, validation_fraction=0.1, activation='relu')
ae.fit(X_ae, X_ae)
# Extract bottleneck: get activations at layer 2 (8-dim)
# MLPRegressor doesn't expose intermediate layers easily, so let's manually forward pass
import copy
def get_bottleneck(model, X):
    """Extract activations at the bottleneck layer"""
    a = X
    for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
        a = a @ w + b
        if i < len(model.coefs_) - 1:
            a = np.maximum(0, a)  # ReLU
        if i == 1:  # bottleneck layer (index 1 = second hidden layer = 8 neurons)
            return a
    return a

X_bottleneck = get_bottleneck(ae, X_ae)
print(f"  Bottleneck shape: {X_bottleneck.shape}")

# Check if latent representations predict PHQ
print("\n  Latent feature PHQ correlations:")
latent_results = []
for i in range(X_bottleneck.shape[1]):
    rho, p = stats.spearmanr(X_bottleneck[:, i], valid_ae['phq_score'])
    latent_results.append({'latent_dim': i, 'rho': round(rho, 4), 'p': p})
    if abs(rho) > 0.05:
        print(f"    Latent_{i}: ρ={rho:.4f}, p={p:.2e}")

# PCA components
print("\n  PCA component PHQ correlations:")
for i in range(8):
    rho, p = stats.spearmanr(X_pca[:, i], valid_ae['phq_score'])
    if abs(rho) > 0.05:
        print(f"    PC{i+1}: ρ={rho:.4f}, p={p:.2e}")

# Compare AUC: raw features vs PCA vs autoencoder latent
y = valid_ae['phq10']
for name, X_test in [('Raw features', X_ae), ('PCA (8d)', X_pca), ('AE bottleneck (8d)', X_bottleneck)]:
    lr = LogisticRegression(class_weight='balanced', C=0.1, max_iter=1000, random_state=42)
    aucs = cross_val_score(lr, X_test, y, cv=5, scoring='roc_auc')
    print(f"  {name}: AUC={aucs.mean():.4f} ± {aucs.std():.3f}")

pd.DataFrame(latent_results).to_csv(f'{OUT}/autoencoder_latent.csv', index=False)

# ============================================================
# ROUND 8: COMPREHENSIVE CROSS-SYSTEM FEATURES
# ============================================================
print("\n" + "="*70)
print("ROUND 8: COMPREHENSIVE CROSS-SYSTEM FEATURES")
print("="*70)

cs_cols = [c for c in feat_cols if any(x in c for x in 
           ['sympathovagal','cardiac_strain','activity_recovery','sleep_rhr',
            'sleep_steps','hrv_steps','rhr_sleep','multi_system','cardiac_recovery',
            'rem_deep','peak_intensity','cardio_intensity','perm_entropy'])]
print(f"  Cross-system features: {len(cs_cols)}")

# Also create interaction terms between top features
top_feats = ['rhr_bpm_mean','sleep_time_minutes_cv','num_steps_mean','rmssd_mean','cardio_minutes_cv']
top_feats = [f for f in top_feats if f in subj.columns]

interaction_results = []
new_interactions = {}
for i, f1 in enumerate(top_feats):
    for f2 in top_feats[i+1:]:
        valid = subj[[f1, f2, 'phq_score']].dropna()
        if len(valid) < 500: continue
        
        # Multiplicative interaction
        inter = valid[f1] * valid[f2]
        inter_name = f'{f1}_x_{f2}'
        rho, p = stats.spearmanr(inter, valid['phq_score'])
        
        # Compare to individual features
        rho1 = stats.spearmanr(valid[f1], valid['phq_score'])[0]
        rho2 = stats.spearmanr(valid[f2], valid['phq_score'])[0]
        
        synergy = abs(rho) > max(abs(rho1), abs(rho2))
        interaction_results.append({
            'interaction': inter_name, 'rho': round(rho, 4), 'p': p,
            'rho_f1': round(rho1, 4), 'rho_f2': round(rho2, 4),
            'synergistic': synergy
        })
        if synergy:
            print(f"  SYNERGY: {inter_name}: ρ={rho:.4f} > max({rho1:.4f}, {rho2:.4f})")
            new_interactions[inter_name] = inter

# Also: ratio features
for i, f1 in enumerate(top_feats):
    for f2 in top_feats[i+1:]:
        valid = subj[[f1, f2, 'phq_score']].dropna()
        if len(valid) < 500: continue
        
        ratio = valid[f1] / (valid[f2] + 1e-10)
        ratio_name = f'{f1}_div_{f2}'
        rho, p = stats.spearmanr(ratio, valid['phq_score'])
        rho1 = stats.spearmanr(valid[f1], valid['phq_score'])[0]
        rho2 = stats.spearmanr(valid[f2], valid['phq_score'])[0]
        synergy = abs(rho) > max(abs(rho1), abs(rho2))
        if synergy:
            print(f"  SYNERGY (ratio): {ratio_name}: ρ={rho:.4f} > max({rho1:.4f}, {rho2:.4f})")
        interaction_results.append({
            'interaction': ratio_name, 'rho': round(rho, 4), 'p': p,
            'rho_f1': round(rho1, 4), 'rho_f2': round(rho2, 4),
            'synergistic': synergy
        })

int_df = pd.DataFrame(interaction_results).sort_values('p')
int_df.to_csv(f'{OUT}/interaction_features.csv', index=False)
n_synergistic = int_df['synergistic'].sum()
print(f"  Total interactions tested: {len(int_df)}")
print(f"  Synergistic (|ρ_inter| > max(|ρ_f1|, |ρ_f2|)): {n_synergistic}")

# ============================================================
# ROUND 9: FULL VALIDATION (temporal + subject split)
# ============================================================
print("\n" + "="*70)
print("ROUND 9: FULL VALIDATION")
print("="*70)

# Subject-level 70/30 split (same as before but with clean data)
np.random.seed(42)
users = subj['user_id'].unique().copy()
np.random.shuffle(users)
split = int(0.7 * len(users))
disc = subj[subj['user_id'].isin(users[:split])].copy()
val = subj[subj['user_id'].isin(users[split:])].copy()
print(f"  Discovery: n={len(disc)}, PHQ10={disc['phq10'].sum()}")
print(f"  Validation: n={len(val)}, PHQ10={val['phq10'].sum()}")

# Feature selection in discovery
screen = []
for f in feat_cols:
    d = disc[[f,'phq_score']].dropna()
    if len(d)<200: continue
    rho, p = stats.spearmanr(d[f], d['phq_score'])
    screen.append({'feature':f, 'rho':rho, 'p':p})
from statsmodels.stats.multitest import multipletests
screen_df = pd.DataFrame(screen)
_, fdr, _, _ = multipletests(screen_df['p'], method='fdr_bh')
screen_df['fdr'] = fdr
disc_sig = screen_df[screen_df['fdr']<0.05]
print(f"  Discovery FDR-sig: {len(disc_sig)}")

# Validate
replicated = 0
for _, row in disc_sig.iterrows():
    f = row['feature']
    v = val[[f,'phq_score']].dropna()
    if len(v)<50: continue
    rho_v, p_v = stats.spearmanr(v[f], v['phq_score'])
    if np.sign(rho_v)==np.sign(row['rho']) and p_v<0.05:
        replicated += 1
print(f"  Replicated: {replicated}/{len(disc_sig)} ({100*replicated/max(1,len(disc_sig)):.0f}%)")

# Model validation
stable = ['rhr_bpm_mean','sleep_time_minutes_cv','cardio_minutes_cv',
          'efficiency_mean','rmssd_mean','peak_minutes_mean','num_steps_mean','rhr_bpm_std',
          'cardiac_strain','peak_intensity_ratio']
stable = [f for f in stable if f in subj.columns]

for outcome, label in [('phq10','Depression'), ('gad10','Anxiety')]:
    X_d = disc[stable].fillna(disc[stable].median())
    X_v = val[stable].fillna(val[stable].median())
    y_d = disc[outcome]; y_v = val[outcome]
    if y_d.isna().any() or y_v.isna().any(): continue
    
    sc = StandardScaler()
    X_ds = sc.fit_transform(X_d)
    X_vs = sc.transform(X_v)
    
    lr = LogisticRegression(class_weight='balanced',C=0.1,max_iter=1000,random_state=42)
    lr.fit(X_ds, y_d)
    auc_d = roc_auc_score(y_d, lr.predict_proba(X_ds)[:,1])
    auc_v = roc_auc_score(y_v, lr.predict_proba(X_vs)[:,1])
    print(f"  {label}: disc AUC={auc_d:.3f}, val AUC={auc_v:.3f}, gap={auc_d-auc_v:.3f}")

# ============================================================
# ROUND 10: INTEGRATION + ABLATION
# ============================================================
print("\n" + "="*70)
print("ROUND 10: INTEGRATION & ABLATION")
print("="*70)

# Full model comparison
valid_all = subj[stable + ['phq10','age','gender_score','financial_situation_score']].dropna()
y = valid_all['phq10']

configs = {
    'Demographics only': ['age','gender_score','financial_situation_score'],
    'Wearables only': stable,
    'Combined': stable + ['age','gender_score','financial_situation_score'],
    'RHR + Sleep CV only': ['rhr_bpm_mean','sleep_time_minutes_cv'],
    'Top 5 wearable': ['rhr_bpm_mean','sleep_time_minutes_cv','cardio_minutes_cv','peak_minutes_mean','cardiac_strain'],
}

print("\n  Model comparison (5-fold CV AUC):")
ablation_results = {}
for name, cols in configs.items():
    cols_use = [c for c in cols if c in valid_all.columns]
    X = StandardScaler().fit_transform(valid_all[cols_use])
    lr = LogisticRegression(class_weight='balanced',C=0.1,max_iter=1000,random_state=42)
    aucs = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
    ablation_results[name] = {'auc_mean': round(aucs.mean(), 4), 'auc_std': round(aucs.std(), 4), 'n_features': len(cols_use)}
    print(f"    {name} ({len(cols_use)} feats): AUC={aucs.mean():.4f} ± {aucs.std():.3f}")

# Same for anxiety
print("\n  Anxiety (GAD-7>=10):")
valid_gad = subj[stable + ['gad10','age','gender_score','financial_situation_score']].dropna()
y_gad = valid_gad['gad10']
for name, cols in configs.items():
    cols_use = [c for c in cols if c in valid_gad.columns]
    X = StandardScaler().fit_transform(valid_gad[cols_use])
    lr = LogisticRegression(class_weight='balanced',C=0.1,max_iter=1000,random_state=42)
    aucs = cross_val_score(lr, X, y_gad, cv=5, scoring='roc_auc')
    print(f"    {name}: AUC={aucs.mean():.4f} ± {aucs.std():.3f}")

# Ceiling analysis (clean)
print("\n  Information ceiling (k-NN vs LR, CV):")
top_by_rho = screen_df.reindex(screen_df['rho'].abs().sort_values(ascending=False).index)
for n_feat in [2, 5, 10, 20, 50]:
    feats = top_by_rho.head(n_feat)['feature'].tolist()
    feats = [f for f in feats if f in valid_all.columns]
    if len(feats) == 0: continue
    X = StandardScaler().fit_transform(valid_all[feats].fillna(valid_all[feats].median()))
    knn = KNeighborsClassifier(n_neighbors=50, weights='distance')
    lr = LogisticRegression(class_weight='balanced',C=0.1,max_iter=1000,random_state=42)
    auc_knn = cross_val_score(knn, X, y, cv=5, scoring='roc_auc').mean()
    auc_lr = cross_val_score(lr, X, y, cv=5, scoring='roc_auc').mean()
    print(f"    {n_feat} feats: k-NN={auc_knn:.4f}, LR={auc_lr:.4f}")

# R² ceiling
X_top = StandardScaler().fit_transform(valid_all[top_by_rho.head(50)['feature'].tolist()[:50]].fillna(0))
y_cont = valid_all['phq10'].map({0: subj.loc[valid_all.index, 'phq_score']}).values
# Actually use continuous
valid_cont = subj[top_by_rho.head(30)['feature'].tolist()[:30] + ['phq_score']].dropna()
X_cont = StandardScaler().fit_transform(valid_cont.drop('phq_score', axis=1))
y_pred = cross_val_predict(KNeighborsRegressor(50, weights='distance'), X_cont, valid_cont['phq_score'], cv=5)
r2 = 1 - np.sum((valid_cont['phq_score'] - y_pred)**2)/np.sum((valid_cont['phq_score'] - valid_cont['phq_score'].mean())**2)
print(f"\n  R² ceiling (30 features, k-NN CV): {r2:.4f}")

# ElasticNet R²
y_pred_en = cross_val_predict(ElasticNet(alpha=0.1, random_state=42), X_cont, valid_cont['phq_score'], cv=5)
r2_en = 1 - np.sum((valid_cont['phq_score'] - y_pred_en)**2)/np.sum((valid_cont['phq_score'] - valid_cont['phq_score'].mean())**2)
print(f"  R² ElasticNet (30 features, CV): {r2_en:.4f}")

print("\n" + "="*70)
print("ALL 10 ROUNDS COMPLETE")
print("="*70)

# Final summary
summary = {
    'round6_dose_response': {d['biomarker']: {'rr_q5q1': d['risk_ratio_q5_q1'], 'nonlinear_p': d['nonlinearity_p']} 
                             for d in dose_results},
    'round7_autoencoder': {'n_latent_sig': sum(1 for r in latent_results if r['p']<0.05)},
    'round8_interactions': {'n_tested': len(int_df), 'n_synergistic': int(n_synergistic)},
    'round9_validation': {'disc_sig': len(disc_sig), 'replicated': replicated},
    'round10_ablation': ablation_results,
    'r2_ceiling_knn': round(r2, 4),
    'r2_elasticnet': round(r2_en, 4)
}
with open(f'{OUT}/r6to10_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
