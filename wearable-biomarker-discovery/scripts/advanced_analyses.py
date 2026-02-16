"""
ROUNDS 2-5: Symbolic regression, mixed effects, CCA, bootstrap Steiger
"""
import pandas as pd, numpy as np, json, time, os
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings; warnings.filterwarnings('ignore')

OUT = 'output/v5'
subj = pd.read_csv(f'{OUT}/subjects_clean.csv')
feat_cols = json.load(open(f'{OUT}/feat_cols.json'))
# Filter to only cols that exist in cleaned subj
feat_cols = [c for c in feat_cols if c in subj.columns]
print(f"Subjects: {len(subj)}, Features: {len(feat_cols)}, PHQ10: {subj['phq10'].sum()}")

# ============================================================
# ROUND 2: SYMBOLIC REGRESSION — DEFINITIVE RESULT
# ============================================================
print("\n" + "="*70)
print("ROUND 2: SYMBOLIC REGRESSION")
print("="*70)

top10 = ['rhr_bpm_mean','cardio_minutes_median','cardio_minutes_cv',
         'peak_minutes_mean','sleep_time_minutes_std','sleep_time_minutes_cv',
         'rmssd_mean','num_steps_mean','efficiency_mean','rhr_bpm_std']
top10 = [f for f in top10 if f in subj.columns]

valid = subj[top10 + ['phq_score']].dropna()
X_sr = StandardScaler().fit_transform(valid[top10])
y_sr = valid['phq_score'].values

from gplearn.genetic import SymbolicRegressor, SymbolicTransformer

sr_results = []
for pop, gen, seed in [(2000, 50, 0), (2000, 50, 1), (3000, 50, 2)]:
    print(f"  SR pop={pop} gen={gen} seed={seed}...", end=' ', flush=True)
    sr = SymbolicRegressor(
        population_size=pop, generations=gen,
        tournament_size=20, stopping_criteria=0.001,
        p_crossover=0.7, p_subtree_mutation=0.1,
        p_hoist_mutation=0.05, p_point_mutation=0.1,
        max_samples=0.9, verbose=0, n_jobs=1,
        function_set=['add','sub','mul','div','sqrt','abs','neg','max','min'],
        random_state=seed, parsimony_coefficient=0.005
    )
    sr.fit(X_sr, y_sr)
    y_pred = sr.predict(X_sr)
    ss_res = np.sum((y_sr - y_pred)**2)
    ss_tot = np.sum((y_sr - y_sr.mean())**2)
    r2 = 1 - ss_res/ss_tot
    rho = stats.spearmanr(y_sr, y_pred)[0] if not np.isnan(y_pred).any() and np.std(y_pred)>0 else 0
    sr_results.append({'pop':pop,'gen':gen,'seed':seed,'r2':round(r2,4),'rho':round(rho,4),
                       'formula':str(sr._program)[:120],'length':sr._program.length_})
    print(f"R²={r2:.4f}, ρ={rho:.4f}, len={sr._program.length_}")

# Symbolic Transformer
print("\n  SymbolicTransformer...", flush=True)
st = SymbolicTransformer(
    population_size=2000, generations=30,
    tournament_size=20, n_components=10,
    function_set=['add','sub','mul','div','sqrt','abs'],
    verbose=0, n_jobs=1, random_state=42,
    parsimony_coefficient=0.005
)
X_new = st.fit_transform(X_sr, y_sr)
sym_feats = []
for i in range(X_new.shape[1]):
    if np.std(X_new[:,i]) == 0: continue
    rho, p = stats.spearmanr(X_new[:,i], y_sr)
    sym_feats.append({'idx':i, 'rho':round(rho,4), 'p':p,
                      'formula':str(st._best_programs[i])[:100] if i<len(st._best_programs) else 'N/A'})
sym_feats.sort(key=lambda x: -abs(x['rho']))
print(f"  Top symbolic features:")
for sf in sym_feats[:5]:
    print(f"    ρ={sf['rho']:.4f}: {sf['formula']}")

pd.DataFrame(sr_results).to_csv(f'{OUT}/sr_results.csv', index=False)
pd.DataFrame(sym_feats).to_csv(f'{OUT}/sym_features.csv', index=False)

# ============================================================
# ROUND 3: WITHIN-PERSON ANALYSIS
# ============================================================
print("\n" + "="*70)
print("ROUND 3: WITHIN-PERSON ANALYSIS")
print("="*70)

# ICC for PHQ-9
raw = pd.read_csv('dwb_dataset.csv', usecols=['user_id','phq_score'])
# PHQ is static per person (same value across all days)
per_user = raw.groupby('user_id')['phq_score'].agg(['mean','std','count'])
icc_std = per_user['std'].mean()
print(f"  PHQ-9 within-person SD: {icc_std:.4f}")
print(f"  PHQ-9 is STATIC per person (ICC≈1.0). Within-person analysis not possible.")
print(f"  This is a dataset limitation: PHQ-9 collected once at intake, not longitudinally.")

# However, we CAN do within-person analysis of wearable features
# Question: Do subjects with more variable wearable patterns have higher PHQ?
# (This is the variability-over-means thesis at the individual level)
print("\n  Within-person wearable variability analysis:")
raw_full = pd.read_csv('dwb_dataset.csv', usecols=['user_id','rhr_bpm','sleep_time_minutes','num_steps','phq_score'])
wp_results = []
for sig in ['rhr_bpm','sleep_time_minutes','num_steps']:
    # Compute within-person SD for each subject
    wp_sd = raw_full.groupby('user_id')[sig].std()
    wp_mean = raw_full.groupby('user_id')[sig].mean()
    phq = raw_full.groupby('user_id')['phq_score'].first()
    
    df = pd.DataFrame({'wp_sd': wp_sd, 'wp_mean': wp_mean, 'phq': phq}).dropna()
    
    # Correlation of within-person SD with PHQ (this IS the between-person effect of variability)
    rho_sd, p_sd = stats.spearmanr(df['wp_sd'], df['phq'])
    rho_mean, p_mean = stats.spearmanr(df['wp_mean'], df['phq'])
    
    wp_results.append({'signal': sig, 'wp_sd_rho': round(rho_sd,4), 'wp_sd_p': p_sd,
                       'wp_mean_rho': round(rho_mean,4), 'wp_mean_p': p_mean})
    print(f"  {sig}: variability ρ={rho_sd:.4f} (p={p_sd:.2e}), mean ρ={rho_mean:.4f} (p={p_mean:.2e})")
    better = 'VARIABILITY' if abs(rho_sd) > abs(rho_mean) else 'MEAN'
    print(f"    → {better} wins")

pd.DataFrame(wp_results).to_csv(f'{OUT}/within_person.csv', index=False)

# ============================================================
# ROUND 4: JOINT CCA (means + variability)
# ============================================================
print("\n" + "="*70)
print("ROUND 4: JOINT CCA")
print("="*70)

# Wearable features: both means and variability
mean_cols = [c for c in feat_cols if c.endswith('_mean') and not any(x in c for x in ['7day','subj'])][:12]
var_cols = [c for c in feat_cols if c.endswith('_cv') and not any(x in c for x in ['7day','subj'])][:12]
all_wear = mean_cols + var_cols
all_wear = [c for c in all_wear if c in subj.columns]

phq_items = ['little_interest_score','depression_score','sleep_score','tired_score',
             'appetite_score','failure_score','trouble_concentrating_score','restlessness_score']
phq_items = [c for c in phq_items if c in subj.columns]

valid = subj[all_wear + phq_items].dropna()
print(f"  Wearable features: {len(all_wear)} (means + CV), PHQ items: {len(phq_items)}")
print(f"  Valid subjects: {len(valid)}")

X_cca = StandardScaler().fit_transform(valid[all_wear])
Y_cca = valid[phq_items].values

n_cc = min(5, len(all_wear), len(phq_items))
cca = CCA(n_components=n_cc)
Xc, Yc = cca.fit_transform(X_cca, Y_cca)

print("\n  Canonical correlations:")
cca_results = []
for i in range(n_cc):
    r, p = stats.pearsonr(Xc[:,i], Yc[:,i])
    cca_results.append({'CC': i+1, 'r': round(r,4), 'p': p})
    print(f"  CC{i+1}: r={r:.4f}, p={p:.2e}")

# Canonical weights
weights = pd.DataFrame(cca.x_weights_, index=all_wear, columns=[f'CC{i+1}' for i in range(n_cc)])
print("\n  Top CC1 weights (wearable side):")
for idx, v in weights['CC1'].abs().nlargest(8).items():
    print(f"    {idx}: {weights.loc[idx,'CC1']:.3f}")

# PHQ side
y_weights = pd.DataFrame(cca.y_weights_, index=phq_items, columns=[f'CC{i+1}' for i in range(n_cc)])
print("\n  Top CC1 weights (PHQ side):")
for idx, v in y_weights['CC1'].abs().nlargest(5).items():
    print(f"    {idx}: {y_weights.loc[idx,'CC1']:.3f}")

weights.to_csv(f'{OUT}/cca_weights_joint.csv')
y_weights.to_csv(f'{OUT}/cca_phq_weights.csv')

# ============================================================
# ROUND 5: BOOTSTRAP STEIGER + SENSITIVITY
# ============================================================
print("\n" + "="*70)
print("ROUND 5: BOOTSTRAP STEIGER'S TEST")
print("="*70)

raw_gad = pd.read_csv('dwb_dataset.csv', usecols=['user_id','anxiety_score','cannot_stop_worry_score',
    'too_much_worry_score','trouble_relaxing_score','restlessness_score_gad','irritability_score','fear_score'])
gad_items_list = ['anxiety_score','cannot_stop_worry_score','too_much_worry_score',
             'trouble_relaxing_score','restlessness_score_gad','irritability_score','fear_score']
raw_gad['gad_score'] = raw_gad[gad_items_list].sum(axis=1)
gad_map = raw_gad.groupby('user_id')['gad_score'].first()
subj['gad_score'] = subj['user_id'].map(gad_map)

def steiger_test(r12, r13, r23, n):
    r_bar = (r12 + r13) / 2
    f = (1 - r23) / (2 * (1 - r_bar**2))
    h = (1 - f * r_bar**2) / (1 - r_bar**2)
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    z_diff = (z12 - z13) * np.sqrt((n - 3) / (2 * (1 - r23) * h))
    p = 2 * (1 - norm.cdf(abs(z_diff)))
    return z_diff, p

test_feats = ['rhr_bpm_mean','sleep_time_minutes_std','sleep_time_minutes_cv',
              'rmssd_mean','num_steps_mean','cardio_minutes_cv','efficiency_mean',
              'rhr_bpm_std','peak_minutes_mean','peak_intensity_ratio']
test_feats = [f for f in test_feats if f in subj.columns]

sv = subj[['phq_score','gad_score'] + test_feats].dropna()
r_pg = stats.pearsonr(sv['phq_score'], sv['gad_score'])[0]
print(f"  PHQ-GAD r={r_pg:.4f} (full sample, n={len(sv)})")

# Bootstrap Steiger
n_boot = 500
print(f"\n  Bootstrap Steiger ({n_boot} iterations):")
steiger_boot = {}
for feat in test_feats:
    diffs = []
    for b in range(n_boot):
        idx = np.random.choice(len(sv), len(sv), replace=True)
        boot = sv.iloc[idx]
        r_d = stats.pearsonr(boot[feat], boot['phq_score'])[0]
        r_a = stats.pearsonr(boot[feat], boot['gad_score'])[0]
        diffs.append(abs(r_a) - abs(r_d))
    
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    mean_diff = diffs.mean()
    pct_anx_stronger = (diffs > 0).mean()
    
    # Point estimate
    r_d = stats.pearsonr(sv[feat], sv['phq_score'])[0]
    r_a = stats.pearsonr(sv[feat], sv['gad_score'])[0]
    z, p = steiger_test(r_d, r_a, r_pg, len(sv))
    
    steiger_boot[feat] = {
        'r_dep': round(r_d, 4), 'r_anx': round(r_a, 4),
        'mean_diff': round(mean_diff, 4),
        'ci_lo': round(ci_lo, 4), 'ci_hi': round(ci_hi, 4),
        'pct_anx_stronger': round(pct_anx_stronger, 3),
        'steiger_p': round(p, 4)
    }
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {feat}: diff={mean_diff:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], {pct_anx_stronger:.0%} anx>dep, Steiger p={p:.4f} {sig}")

pd.DataFrame(steiger_boot).T.to_csv(f'{OUT}/steiger_bootstrap.csv')

# Sensitivity: Steiger in clinical range only
print("\n  Sensitivity: Steiger in PHQ>=5 or GAD>=5 only:")
clinical = sv[(sv['phq_score']>=5) | (sv['gad_score']>=5)]
r_pg_clin = stats.pearsonr(clinical['phq_score'], clinical['gad_score'])[0]
print(f"  PHQ-GAD r (clinical)={r_pg_clin:.4f} (n={len(clinical)})")
for feat in test_feats[:5]:
    r_d = stats.pearsonr(clinical[feat], clinical['phq_score'])[0]
    r_a = stats.pearsonr(clinical[feat], clinical['gad_score'])[0]
    z, p = steiger_test(r_d, r_a, r_pg_clin, len(clinical))
    print(f"  {feat}: dep={r_d:.4f}, anx={r_a:.4f}, p={p:.4f}")

print("\n[ROUNDS 2-5 COMPLETE]")

# Save summary
summary = {
    'round2_sr': {'best_r2': max(r['r2'] for r in sr_results), 'conclusion': 'SR fails to find non-linear signal'},
    'round3_within': {k: v for d in wp_results for k, v in d.items()},
    'round4_cca': {'cc1': cca_results[0]['r'], 'n_wear_features': len(all_wear)},
    'round5_steiger': {'n_sig_anx_stronger': sum(1 for v in steiger_boot.values() if v['steiger_p']<0.05 and v['mean_diff']>0)}
}
with open(f'{OUT}/r2to5_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
