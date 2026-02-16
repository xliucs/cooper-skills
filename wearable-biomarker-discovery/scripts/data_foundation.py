"""
ROUND 1: Clean data foundation. Fix all leakage, build proper feature set.
"""
import pandas as pd, numpy as np, json, os, time
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import warnings; warnings.filterwarnings('ignore')

OUT = 'output/v5'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

print("="*70)
print("ROUND 1: CLEAN DATA FOUNDATION")
print("="*70)

# ============================================================
# 1. Load raw data, compute ALL outcomes
# ============================================================
print("\n[1.1] Loading raw data...")
raw = pd.read_csv('dwb_dataset.csv')
print(f"  Raw: {raw.shape[0]:,} rows, {raw['user_id'].nunique():,} subjects")

# GAD-7
gad_items = ['anxiety_score','cannot_stop_worry_score','too_much_worry_score',
             'trouble_relaxing_score','restlessness_score_gad','irritability_score','fear_score']
raw['gad_score'] = raw[gad_items].sum(axis=1)

# PHQ-9 items
phq_items_raw = ['little_interest_score','depression_score','sleep_score','tired_score',
                 'appetite_score','failure_score','trouble_concentrating_score','restlessness_score']

# BFI-10 personality
personality = ['neuroticism_score','extraversion_score','agreeableness_score',
               'conscientiousness_score','openness_score']

# Demographics + confounders
demographics = ['age','gender_score','financial_situation_score']

# ============================================================
# 1.2 Subject-level aggregation — CLEAN
# ============================================================
print("\n[1.2] Building clean subject-level features...")

# Core wearable signals (raw daily measures)
wearable_signals = ['rmssd','rhr_bpm','rate_brpm','num_steps','efficiency',
                    'sleep_time_minutes','deep_sleep_minutes','rem_sleep_percent',
                    'awake_minutes','restlessness','waso_count_long_wakes',
                    'cardio_minutes','fat_burn_minutes','peak_minutes',
                    'overall_score','revitalization_score','duration_minutes']

# Pre-computed 7-day rolling features in raw data
rolling_cols = [c for c in raw.columns if '_7day_' in c]

features = {}
for uid, grp in raw.groupby('user_id'):
    if len(grp) < 14:
        continue
    
    row = {'user_id': uid, 'n_days': len(grp)}
    
    # Static outcomes (first row)
    row['phq_score'] = grp['phq_score'].iloc[0]
    row['gad_score'] = grp['gad_score'].iloc[0]
    for item in phq_items_raw:
        row[item] = grp[item].iloc[0]
    
    # Demographics
    for d in demographics:
        row[d] = grp[d].iloc[0]
    
    # Personality
    for p in personality:
        if p in grp.columns:
            row[p] = grp[p].iloc[0]
    
    # Wearable features — comprehensive per-signal aggregation
    for sig in wearable_signals:
        vals = grp[sig].dropna()
        if len(vals) < 7:
            continue
        row[f'{sig}_mean'] = vals.mean()
        row[f'{sig}_std'] = vals.std()
        row[f'{sig}_cv'] = vals.std() / vals.mean() if vals.mean() != 0 else np.nan
        row[f'{sig}_median'] = vals.median()
        row[f'{sig}_iqr'] = vals.quantile(0.75) - vals.quantile(0.25)
        row[f'{sig}_skew'] = vals.skew()
        row[f'{sig}_min'] = vals.min()
        row[f'{sig}_max'] = vals.max()
        row[f'{sig}_range'] = vals.max() - vals.min()
        
        # Temporal features
        if len(vals) >= 3:
            diffs = vals.diff().dropna()
            row[f'{sig}_daily_change_mean'] = diffs.abs().mean()
            row[f'{sig}_daily_change_std'] = diffs.std()
            row[f'{sig}_autocorr'] = vals.autocorr() if len(vals) >= 5 else np.nan
            # Trend (slope)
            x = np.arange(len(vals))
            slope, _, _, _, _ = stats.linregress(x, vals)
            row[f'{sig}_trend'] = slope
        
        # Weekend vs weekday
        if 'weekday' in grp.columns:
            wkday = vals[grp.loc[vals.index, 'weekday'] < 5] if 'weekday' in grp.columns else vals
            wkend = vals[grp.loc[vals.index, 'weekday'] >= 5] if 'weekday' in grp.columns else vals
            if len(wkday) >= 3 and len(wkend) >= 1:
                row[f'{sig}_wkend_diff'] = wkend.mean() - wkday.mean()
    
    # Rolling feature aggregations (7-day stats of stats)
    for rc in rolling_cols:
        vals = grp[rc].dropna()
        if len(vals) >= 5:
            row[f'{rc}_subj_mean'] = vals.mean()
            row[f'{rc}_subj_std'] = vals.std()
    
    # Cross-system features
    if all(f'{s}_mean' in row for s in ['rhr_bpm','rmssd','sleep_time_minutes','num_steps']):
        # Sympathovagal balance
        if row.get('rmssd_mean', 0) > 0:
            row['sympathovagal_ratio'] = row['rhr_bpm_mean'] / row['rmssd_mean']
        
        # Cardiac strain (RHR std * RHR mean)
        row['cardiac_strain'] = row.get('rhr_bpm_std', 0) * row.get('rhr_bpm_mean', 0)
        
        # Activity-recovery mismatch
        if row.get('rmssd_mean', 0) > 0:
            row['activity_recovery_mismatch'] = row.get('num_steps_mean', 0) / row.get('rmssd_mean', 1)
        
        # Sleep-cardiac coupling: correlation between daily sleep and daily rhr
        sleep_daily = grp['sleep_time_minutes'].dropna()
        rhr_daily = grp['rhr_bpm'].dropna()
        common_idx = sleep_daily.index.intersection(rhr_daily.index)
        if len(common_idx) >= 7:
            row['sleep_rhr_corr'] = stats.pearsonr(sleep_daily[common_idx], rhr_daily[common_idx])[0]
        
        # Sleep-activity coupling
        steps_daily = grp['num_steps'].dropna()
        common_idx = sleep_daily.index.intersection(steps_daily.index)
        if len(common_idx) >= 7:
            row['sleep_steps_corr'] = stats.pearsonr(sleep_daily[common_idx], steps_daily[common_idx])[0]
        
        # HRV-activity coupling
        rmssd_daily = grp['rmssd'].dropna()
        common_idx = rmssd_daily.index.intersection(steps_daily.index)
        if len(common_idx) >= 7:
            row['hrv_steps_corr'] = stats.pearsonr(rmssd_daily[common_idx], steps_daily[common_idx])[0]
        
        # RHR-sleep coupling
        common_idx = rhr_daily.index.intersection(sleep_daily.index)
        if len(common_idx) >= 7:
            row['rhr_sleep_corr'] = stats.pearsonr(rhr_daily[common_idx], sleep_daily[common_idx])[0]
        
        # Multi-system coherence: mean of abs correlations across all pairs
        pairs_corrs = [abs(row.get('sleep_rhr_corr', 0)), abs(row.get('sleep_steps_corr', 0)),
                       abs(row.get('hrv_steps_corr', 0)), abs(row.get('rhr_sleep_corr', 0))]
        valid_corrs = [c for c in pairs_corrs if not np.isnan(c)]
        if valid_corrs:
            row['multi_system_coherence'] = np.mean(valid_corrs)
        
        # Cardiac recovery efficiency: RMSSD gain per hour of sleep
        if row.get('sleep_time_minutes_mean', 0) > 0:
            row['cardiac_recovery_eff'] = row.get('rmssd_mean', 0) / (row['sleep_time_minutes_mean'] / 60)
        
        # Sleep architecture ratio: REM% / deep%
        if row.get('deep_sleep_minutes_mean', 0) > 0 and row.get('duration_minutes_mean', 0) > 0:
            deep_pct = row['deep_sleep_minutes_mean'] / row['duration_minutes_mean']
            rem_pct = row.get('rem_sleep_percent_mean', 0) / 100
            if deep_pct > 0:
                row['rem_deep_ratio'] = rem_pct / deep_pct
        
        # Activity intensity ratio: peak / (peak + cardio + fat_burn)
        total_active = row.get('peak_minutes_mean', 0) + row.get('cardio_minutes_mean', 0) + row.get('fat_burn_minutes_mean', 0)
        if total_active > 0:
            row['peak_intensity_ratio'] = row.get('peak_minutes_mean', 0) / total_active
            row['cardio_intensity_ratio'] = row.get('cardio_minutes_mean', 0) / total_active
    
    # Permutation entropy (simplified — on daily RHR series)
    rhr_series = grp['rhr_bpm'].dropna().values
    if len(rhr_series) >= 10:
        # Order-3 permutation entropy
        m = 3
        perms = []
        for i in range(len(rhr_series) - m + 1):
            perms.append(tuple(np.argsort(rhr_series[i:i+m])))
        from collections import Counter
        counts = Counter(perms)
        probs = np.array(list(counts.values())) / sum(counts.values())
        row['rhr_perm_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
    
    sleep_series = grp['sleep_time_minutes'].dropna().values
    if len(sleep_series) >= 10:
        m = 3
        perms = []
        for i in range(len(sleep_series) - m + 1):
            perms.append(tuple(np.argsort(sleep_series[i:i+m])))
        from collections import Counter
        counts = Counter(perms)
        probs = np.array(list(counts.values())) / sum(counts.values())
        row['sleep_perm_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
    
    features[uid] = row

subj = pd.DataFrame(features.values())
print(f"  Subjects: {len(subj)}")
print(f"  Features: {len(subj.columns)}")

# ============================================================
# 1.3 Define CLEAN feature columns (no leakage)
# ============================================================
# Strictly exclude: user_id, outcomes, demographics, PHQ items, personality
leakage_cols = {'user_id','phq_score','gad_score','n_days',
                'age','gender_score','financial_situation_score',
                'neuroticism_score','extraversion_score','agreeableness_score',
                'conscientiousness_score','openness_score'} | set(phq_items_raw)

feat_cols = [c for c in subj.columns if c not in leakage_cols 
             and subj[c].dtype in ['float64','int64','float32']
             and subj[c].notna().sum() > len(subj)*0.5  # require 50% non-missing
             and not np.isinf(subj[c]).any()]
print(f"  Clean feature columns: {len(feat_cols)}")

# Binary outcomes
subj['phq10'] = (subj['phq_score'] >= 10).astype(int)
subj['phq5'] = (subj['phq_score'] >= 5).astype(int)
subj['gad10'] = (subj['gad_score'] >= 10).astype(int)

# ============================================================
# 1.4 Global screening
# ============================================================
print("\n[1.3] Global FDR screening (PHQ continuous)...")
screening = []
for f in feat_cols:
    v = subj[[f, 'phq_score']].dropna()
    if len(v) < 500:
        continue
    rho, p = stats.spearmanr(v[f], v['phq_score'])
    screening.append({'feature': f, 'rho': rho, 'p': p, 'n': len(v)})

screen_df = pd.DataFrame(screening)
_, fdr_p, _, _ = multipletests(screen_df['p'], method='fdr_bh')
screen_df['p_fdr'] = fdr_p
screen_df['sig'] = fdr_p < 0.05
screen_df = screen_df.sort_values('p')
screen_df.to_csv(f'{OUT}/screening_clean.csv', index=False)

n_sig = screen_df['sig'].sum()
print(f"  Tested: {len(screen_df)}")
print(f"  FDR-significant: {n_sig}")
print(f"  Top 10:")
for _, r in screen_df.head(10).iterrows():
    print(f"    {r['feature']}: ρ={r['rho']:.4f}, p_fdr={r['p_fdr']:.2e}")

# ============================================================
# 1.5 Novel cross-system features check
# ============================================================
print("\n[1.4] Cross-system feature screening...")
cs_feats = [c for c in feat_cols if any(x in c for x in 
            ['sympathovagal','cardiac_strain','activity_recovery','sleep_rhr_corr',
             'sleep_steps_corr','hrv_steps_corr','rhr_sleep_corr','multi_system',
             'cardiac_recovery_eff','rem_deep_ratio','peak_intensity','cardio_intensity',
             'rhr_perm_entropy','sleep_perm_entropy'])]
for f in cs_feats:
    v = subj[[f,'phq_score']].dropna()
    if len(v) < 500: continue
    rho, p = stats.spearmanr(v[f], v['phq_score'])
    sig = screen_df[screen_df['feature']==f]['sig'].values
    sig_str = '***' if (len(sig)>0 and sig[0]) else ''
    print(f"  {f}: ρ={rho:.4f}, p={p:.4e} {sig_str}")

# Save
subj.to_csv(f'{OUT}/subjects_clean.csv', index=False)

elapsed = time.time() - t0
print(f"\n[ROUND 1 COMPLETE] {elapsed:.0f}s, {len(subj)} subjects, {len(feat_cols)} features")

# Save feature list
with open(f'{OUT}/feat_cols.json', 'w') as f:
    json.dump(feat_cols, f)
with open(f'{OUT}/r1_summary.json', 'w') as f:
    json.dump({
        'n_subjects': len(subj),
        'n_features': len(feat_cols),
        'n_fdr_sig': int(n_sig),
        'phq10_n': int(subj['phq10'].sum()),
        'gad10_n': int(subj['gad10'].sum()),
        'top5': screen_df.head(5)[['feature','rho','p_fdr']].to_dict('records'),
        'n_cross_system': len(cs_feats),
        'elapsed_sec': round(elapsed, 1)
    }, f, indent=2)
