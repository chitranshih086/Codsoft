# ================================================================
#   CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
#   Data Science Internship Project
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    average_precision_score
)
from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline       import Pipeline as ImbPipeline

# ── Global Style ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#080c14',
    'axes.facecolor':   '#0e1521',
    'axes.edgecolor':   '#1e3050',
    'axes.labelcolor':  '#c8d8f0',
    'xtick.color':      '#7a9bc0',
    'ytick.color':      '#7a9bc0',
    'text.color':       '#c8d8f0',
    'grid.color':       '#142030',
    'grid.alpha':       0.7,
    'font.family':      'monospace',
})

BG     = '#080c14'
PANEL  = '#0e1521'
GREEN  = '#00ff88'    # legitimate / safe
RED    = '#ff3860'    # fraud / danger
BLUE   = '#3d9df5'
AMBER  = '#ffb347'
PURPLE = '#c084fc'
CYAN   = '#22d3ee'
WHITE  = '#e8f0ff'

# ================================================================
# 1. LOAD & EXPLORE DATA
# ================================================================
print("=" * 65)
print("   CREDIT CARD FRAUD DETECTION  —  DATA SCIENCE PROJECT")
print("=" * 65)

df = pd.read_csv('/home/claude/creditcard.csv')

print(f"\n📊 Dataset Shape     : {df.shape}")
print(f"📋 Features          : {list(df.columns)}")
print(f"\n🔍 Missing Values    : {df.isnull().sum().sum()}")
print(f"\n💳 Class Distribution:")
vc = df['Class'].value_counts()
print(f"   Legitimate (0)   : {vc[0]:,}  ({vc[0]/len(df)*100:.2f}%)")
print(f"   Fraudulent (1)   : {vc[1]:,}  ({vc[1]/len(df)*100:.2f}%)")
print(f"   Imbalance Ratio  : {vc[0]//vc[1]}:1")

print(f"\n💰 Transaction Amount Stats:")
print(f"   Legitimate  → Mean: ${df[df.Class==0]['Amount'].mean():.2f}"
      f"  Max: ${df[df.Class==0]['Amount'].max():.2f}")
print(f"   Fraudulent  → Mean: ${df[df.Class==1]['Amount'].mean():.2f}"
      f"  Max: ${df[df.Class==1]['Amount'].max():.2f}")

# ================================================================
# 2. EDA VISUALIZATION
# ================================================================
fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle('🔐  CREDIT CARD FRAUD DETECTION  —  EXPLORATORY ANALYSIS',
             fontsize=16, color=RED, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

# Panel 1: Class Distribution (donut)
ax1 = fig.add_subplot(gs[0, 0])
sizes  = [vc[0], vc[1]]
colors = [GREEN, RED]
wedge_props = dict(width=0.55, edgecolor=BG, linewidth=2.5)
wedges, texts, autotexts = ax1.pie(
    sizes, colors=colors, autopct='%1.2f%%',
    startangle=90, wedgeprops=wedge_props,
    pctdistance=0.78,
    textprops={'fontsize': 10, 'color': WHITE})
for at in autotexts:
    at.set_fontweight('bold')
ax1.set_title('Class Distribution', color=AMBER, fontsize=12, pad=10)
legend_patches = [mpatches.Patch(color=GREEN, label=f'Legitimate ({vc[0]:,})'),
                  mpatches.Patch(color=RED,   label=f'Fraudulent ({vc[1]:,})')]
ax1.legend(handles=legend_patches, loc='lower center',
           fontsize=8, framealpha=0.2)

# Panel 2: Amount distribution by class
ax2 = fig.add_subplot(gs[0, 1])
legit_amt = df[df.Class==0]['Amount']
fraud_amt = df[df.Class==1]['Amount']
ax2.hist(legit_amt.clip(upper=500), bins=40, color=GREEN, alpha=0.7,
         label='Legitimate', edgecolor=BG, linewidth=0.3, density=True)
ax2.hist(fraud_amt.clip(upper=500), bins=20, color=RED, alpha=0.85,
         label='Fraudulent', edgecolor=BG, linewidth=0.3, density=True)
ax2.set_title('Transaction Amount Distribution', color=CYAN, fontsize=11, pad=8)
ax2.set_xlabel('Amount ($, clipped at 500)');  ax2.set_ylabel('Density')
ax2.legend(fontsize=9);  ax2.grid(True, alpha=0.35)

# Panel 3: Time distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df[df.Class==0]['Time']/3600, bins=40, color=GREEN,
         alpha=0.7, label='Legitimate', density=True,
         edgecolor=BG, linewidth=0.3)
ax3.hist(df[df.Class==1]['Time']/3600, bins=20, color=RED,
         alpha=0.85, label='Fraudulent', density=True,
         edgecolor=BG, linewidth=0.3)
ax3.set_title('Transaction Time Distribution', color=CYAN, fontsize=11, pad=8)
ax3.set_xlabel('Hours');  ax3.set_ylabel('Density')
ax3.legend(fontsize=9);  ax3.grid(True, alpha=0.35)

# Panel 4–5: V1 & V3 distributions (most discriminative features)
for idx, feat in enumerate(['V1', 'V3']):
    ax = fig.add_subplot(gs[1, idx])
    ax.hist(df[df.Class==0][feat], bins=40, color=GREEN, alpha=0.75,
            label='Legitimate', density=True, edgecolor=BG, linewidth=0.3)
    ax.hist(df[df.Class==1][feat], bins=20, color=RED, alpha=0.85,
            label='Fraudulent', density=True, edgecolor=BG, linewidth=0.3)
    ax.set_title(f'{feat} Feature Distribution', color=PURPLE, fontsize=11, pad=8)
    ax.set_xlabel(feat);  ax.set_ylabel('Density')
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.35)

# Panel 6: Correlation heatmap — fraud vs legit avg feature diff
ax6 = fig.add_subplot(gs[1, 2])
feat_cols = [c for c in df.columns if c not in ['Time','Amount','Class']]
diff = (df[df.Class==1][feat_cols].mean() - df[df.Class==0][feat_cols].mean())
clrs = [RED if v < 0 else GREEN for v in diff.values]
ax6.barh(diff.index, diff.values, color=clrs, alpha=0.85,
         edgecolor=BG, linewidth=0.4)
ax6.axvline(0, color=WHITE, linewidth=1.5, linestyle='--', alpha=0.6)
ax6.set_title('Feature Mean Diff\n(Fraud − Legitimate)', color=AMBER, fontsize=11, pad=8)
ax6.set_xlabel('Δ Mean Value');  ax6.grid(True, alpha=0.3, axis='x')
ax6.tick_params(labelsize=7)

# Panel 7: V1 vs V3 scatter
ax7 = fig.add_subplot(gs[2, :2])
ax7.scatter(df[df.Class==0]['V1'], df[df.Class==0]['V3'],
            c=GREEN, alpha=0.35, s=12, label='Legitimate', zorder=2)
ax7.scatter(df[df.Class==1]['V1'], df[df.Class==1]['V3'],
            c=RED, alpha=0.85, s=40, marker='X', label='Fraudulent', zorder=3,
            edgecolors='white', linewidths=0.4)
ax7.set_title('V1 vs V3 Feature Space  (X = Fraud)', color=RED, fontsize=12, pad=8)
ax7.set_xlabel('V1 (PCA Component)');  ax7.set_ylabel('V3 (PCA Component)')
ax7.legend(fontsize=9, markerscale=1.5);  ax7.grid(True, alpha=0.3)

# Panel 8: Boxplot amount by class
ax8 = fig.add_subplot(gs[2, 2])
bp = ax8.boxplot(
    [legit_amt.clip(upper=800), fraud_amt.clip(upper=800)],
    labels=['Legitimate', 'Fraudulent'], patch_artist=True,
    medianprops=dict(color='white', linewidth=2.5),
    notch=True)
for patch, clr in zip(bp['boxes'], [GREEN, RED]):
    patch.set_facecolor(clr);  patch.set_alpha(0.7)
for el in ['whiskers', 'caps']:
    for item in bp[el]:
        item.set_color('#446688')
ax8.set_title('Amount Boxplot by Class', color=BLUE, fontsize=11, pad=8)
ax8.set_ylabel('Transaction Amount ($)');  ax8.grid(True, alpha=0.3, axis='y')

plt.savefig('/home/claude/eda_fraud.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\n✅  EDA chart saved.")

# ================================================================
# 3. PREPROCESSING
# ================================================================
print("\n" + "=" * 65)
print("   PREPROCESSING & HANDLING CLASS IMBALANCE")
print("=" * 65)

X = df.drop('Class', axis=1).copy()
y = df['Class'].copy()

# Scale Amount and Time
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time']   = scaler.fit_transform(X[['Time']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain set  : {X_train.shape[0]} samples")
print(f"Test  set  : {X_test.shape[0]}  samples")
print(f"\nClass balance in train:")
tc = y_train.value_counts()
print(f"  Legitimate : {tc[0]}  Fraudulent : {tc[1]}")

# ── 3 Sampling Strategies ───────────────────────────────────
print("\n🔧 Applying class-balancing techniques...")

# SMOTE oversampling
sm = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = sm.fit_resample(X_train, y_train)
print(f"  [SMOTE]      Legitimate: {sum(y_smote==0):,}  "
      f"Fraudulent: {sum(y_smote==1):,}")

# Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
print(f"  [Undersample] Legitimate: {sum(y_under==0):,}  "
      f"Fraudulent: {sum(y_under==1):,}")

# Combined (SMOTE + Undersampling)
# Combined: first SMOTE to 0.5 ratio, then undersample majority
sm_comb  = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.5)
X_sm_c, y_sm_c = sm_comb.fit_resample(X_train, y_train)
rus_comb = RandomUnderSampler(random_state=42, sampling_strategy=1.0)
X_comb, y_comb = rus_comb.fit_resample(X_sm_c, y_sm_c)
print(f"  [Combined]   Legitimate: {sum(y_comb==0):,}  "
      f"Fraudulent: {sum(y_comb==1):,}")

# ================================================================
# 4. MODEL TRAINING
# ================================================================
print("\n" + "=" * 65)
print("   MODEL TRAINING — MULTIPLE ALGORITHMS × 3 SAMPLING STRATEGIES")
print("=" * 65)

datasets = {
    'Original'   : (X_train, y_train),
    'SMOTE'      : (X_smote, y_smote),
    'Undersample': (X_under, y_under),
    'Combined'   : (X_comb,  y_comb),
}

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42,
                                               class_weight='balanced'),
    'Random Forest'      : RandomForestClassifier(n_estimators=150, max_depth=8,
                                                   random_state=42, class_weight='balanced'),
    'Gradient Boosting'  : GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                        learning_rate=0.1, random_state=42),
}

all_results = {}

print(f"\n{'Strategy':<13} {'Model':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ROC-AUC':>8}")
print("─" * 65)

for strat_name, (Xtr, ytr) in datasets.items():
    all_results[strat_name] = {}
    for mdl_name, clf in classifiers.items():
        clf_copy = type(clf)(**clf.get_params())
        clf_copy.fit(Xtr, ytr)
        y_pred    = clf_copy.predict(X_test)
        y_proba   = clf_copy.predict_proba(X_test)[:, 1]

        prec    = precision_score(y_test, y_pred, zero_division=0)
        rec     = recall_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        all_results[strat_name][mdl_name] = {
            'clf': clf_copy, 'pred': y_pred, 'proba': y_proba,
            'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc_auc
        }
        print(f"  {strat_name:<12} {mdl_name:<22} "
              f"{prec:>6.4f} {rec:>6.4f} {f1:>6.4f} {roc_auc:>8.4f}")
    print()

# ── Best combination ────────────────────────────────────────
best_f1    = 0
best_combo = ('', '')
for strat, models in all_results.items():
    for mdl_name, res in models.items():
        if res['f1'] > best_f1:
            best_f1    = res['f1']
            best_combo = (strat, mdl_name)

best_strat, best_model = best_combo
best_res = all_results[best_strat][best_model]
print(f"\n🏆 Best Combo : {best_strat}  +  {best_model}")
print(f"   F1={best_f1:.4f}  ROC-AUC={best_res['roc_auc']:.4f}")

print(f"\n📋 Full Classification Report ({best_strat} + {best_model}):")
print(classification_report(y_test, best_res['pred'],
      target_names=['Legitimate', 'Fraudulent']))

# ================================================================
# 5. RESULTS VISUALIZATION
# ================================================================
fig2 = plt.figure(figsize=(20, 18), facecolor=BG)
fig2.suptitle('🔐  FRAUD DETECTION  —  MODEL RESULTS & EVALUATION',
              fontsize=16, color=RED, fontweight='bold', y=0.98)
gs2 = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.52, wspace=0.4)

# Panel 1: Confusion Matrix — Best Model
ax1 = fig2.add_subplot(gs2[0, 0])
cm = confusion_matrix(y_test, best_res['pred'])
sns.heatmap(cm, annot=True, fmt='d', ax=ax1,
            cmap='RdYlGn', linewidths=2, linecolor=BG,
            annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'},
            xticklabels=['Predicted\nLegit', 'Predicted\nFraud'],
            yticklabels=['Actual\nLegit', 'Actual\nFraud'])
ax1.set_title(f'Confusion Matrix\n[{best_model}]', color=AMBER, fontsize=11, pad=8)
tn, fp, fn, tp = cm.ravel()
ax1.text(0.5, -0.28, f'TN={tn}  FP={fp}  FN={fn}  TP={tp}',
         ha='center', transform=ax1.transAxes, fontsize=8, color='#7a9bc0')

# Panel 2: ROC Curves — all models with best strategy
ax2 = fig2.add_subplot(gs2[0, 1])
roc_colors = [BLUE, GREEN, PURPLE]
for (mdl_name, res), clr in zip(all_results[best_strat].items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    ax2.plot(fpr, tpr, color=clr, linewidth=2.5,
             label=f"{mdl_name.split()[0]} (AUC={res['roc_auc']:.3f})")
ax2.plot([0,1],[0,1], '--', color='#445566', linewidth=1.5, label='Random Baseline')
ax2.fill_between([0,1],[0,1], alpha=0.05, color='#445566')
ax2.set_title(f'ROC Curves  [{best_strat}]', color=CYAN, fontsize=11, pad=8)
ax2.set_xlabel('False Positive Rate');  ax2.set_ylabel('True Positive Rate')
ax2.legend(fontsize=8, loc='lower right');  ax2.grid(True, alpha=0.3)

# Panel 3: Precision-Recall Curve
ax3 = fig2.add_subplot(gs2[0, 2])
for (mdl_name, res), clr in zip(all_results[best_strat].items(), roc_colors):
    prec_c, rec_c, _ = precision_recall_curve(y_test, res['proba'])
    ap = average_precision_score(y_test, res['proba'])
    ax3.plot(rec_c, prec_c, color=clr, linewidth=2.5,
             label=f"{mdl_name.split()[0]} (AP={ap:.3f})")
baseline = y_test.mean()
ax3.axhline(baseline, linestyle='--', color='#445566', linewidth=1.5,
            label=f'Baseline ({baseline:.3f})')
ax3.set_title('Precision-Recall Curves', color=PURPLE, fontsize=11, pad=8)
ax3.set_xlabel('Recall');  ax3.set_ylabel('Precision')
ax3.legend(fontsize=8);  ax3.grid(True, alpha=0.3)

# Panel 4: F1 Score Heatmap across strategies × models
ax4 = fig2.add_subplot(gs2[1, :2])
strat_names = list(all_results.keys())
mdl_names   = list(classifiers.keys())
f1_matrix   = np.array([[all_results[s][m]['f1'] for m in mdl_names] for s in strat_names])
im = ax4.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
plt.colorbar(im, ax=ax4, shrink=0.85)
for i in range(len(strat_names)):
    for j in range(len(mdl_names)):
        ax4.text(j, i, f'{f1_matrix[i,j]:.3f}',
                 ha='center', va='center', fontsize=11, fontweight='bold',
                 color='black' if f1_matrix[i,j] > 0.4 else 'white')
ax4.set_xticks(range(len(mdl_names)));   ax4.set_xticklabels(mdl_names, fontsize=9)
ax4.set_yticks(range(len(strat_names))); ax4.set_yticklabels(strat_names, fontsize=9)
ax4.set_title('F1-Score Heatmap  (Strategy × Model)', color=GREEN, fontsize=12, pad=8)

# Panel 5: Feature Importance
ax5 = fig2.add_subplot(gs2[1, 2])
rf_mdl    = all_results[best_strat]['Random Forest']['clf']
importances = pd.Series(rf_mdl.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True).tail(12)
bar_clrs  = [RED if v > importances.median() else BLUE for v in importances.values]
ax5.barh(importances.index, importances.values,
         color=bar_clrs, alpha=0.9, edgecolor=BG, linewidth=0.4)
ax5.set_title('Feature Importance\n(Random Forest)', color=AMBER, fontsize=11, pad=8)
ax5.set_xlabel('Importance');  ax5.grid(True, alpha=0.3, axis='x')
ax5.tick_params(labelsize=8)

# Panel 6: Metrics comparison bar chart
ax6 = fig2.add_subplot(gs2[2, :2])
metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metric_names))
width = 0.22
bar_colors = [BLUE, GREEN, PURPLE]
for i, (mdl_name, clr) in enumerate(zip(mdl_names, bar_colors)):
    res  = all_results[best_strat][mdl_name]
    vals = [res['precision'], res['recall'], res['f1'], res['roc_auc']]
    bars = ax6.bar(x + i*width, vals, width,
                   color=clr, alpha=0.85, edgecolor=BG, linewidth=0.4,
                   label=mdl_name.split()[0])
    for bar, val in zip(bars, vals):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7, color='white')
ax6.set_xticks(x + width);  ax6.set_xticklabels(metric_names, fontsize=10)
ax6.set_ylim(0, 1.15);  ax6.set_ylabel('Score')
ax6.set_title(f'Metrics Comparison  [{best_strat}]', color=CYAN, fontsize=12, pad=8)
ax6.legend(fontsize=9);  ax6.grid(True, alpha=0.3, axis='y')
ax6.axhline(0.9, color=GREEN, linewidth=1.5, linestyle=':', alpha=0.5)

# Panel 7: Performance Dashboard
ax7 = fig2.add_subplot(gs2[2, 2])
ax7.axis('off')
metrics_display = [
    ('F1-Score',   f'{best_res["f1"]:.4f}',       GREEN),
    ('ROC-AUC',    f'{best_res["roc_auc"]:.4f}',   CYAN),
    ('Precision',  f'{best_res["precision"]:.4f}',  BLUE),
    ('Recall',     f'{best_res["recall"]:.4f}',     PURPLE),
]
ax7.text(0.5, 0.97, 'Best Model Performance', ha='center', va='top',
         transform=ax7.transAxes, fontsize=10, color=AMBER, fontweight='bold')
ax7.text(0.5, 0.88, f'{best_strat} + {best_model}', ha='center', va='top',
         transform=ax7.transAxes, fontsize=8, color='#7a9bc0')
for i, (label, val, clr) in enumerate(metrics_display):
    yp = 0.72 - i*0.19
    rect = plt.Rectangle((0.05, yp-0.07), 0.9, 0.14,
                          fill=True, facecolor='#0e1521',
                          edgecolor=clr, linewidth=1.5,
                          transform=ax7.transAxes)
    ax7.add_patch(rect)
    ax7.text(0.5, yp+0.03, val, ha='center', va='center',
             transform=ax7.transAxes, fontsize=17,
             color=clr, fontweight='bold')
    ax7.text(0.5, yp-0.04, label, ha='center', va='center',
             transform=ax7.transAxes, fontsize=8, color='#6a8ab0')

# Fraud caught summary
caught = tp;  missed = fn;  false_alarm = fp
ax7.text(0.5, 0.01,
         f'✅ Caught: {caught}  ❌ Missed: {missed}  ⚠ False Alarms: {false_alarm}',
         ha='center', va='bottom', transform=ax7.transAxes,
         fontsize=7.5, color='#7a9bc0')

plt.savefig('/home/claude/model_results_fraud.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("✅  Model results chart saved.")

# ================================================================
# 6. SAMPLE FRAUD PREDICTIONS
# ================================================================
print("\n" + "=" * 65)
print("   SAMPLE TRANSACTION PREDICTIONS")
print("=" * 65)

best_clf = all_results[best_strat][best_model]['clf']

sample_idx = X_test.index[:8]
sample_X   = X_test.loc[sample_idx]
sample_y   = y_test.loc[sample_idx]
sample_pred  = best_clf.predict(sample_X)
sample_proba = best_clf.predict_proba(sample_X)[:, 1]

print(f"\n  {'#':<4} {'Actual':<14} {'Predicted':<14} {'Fraud Prob':>10}  {'Status'}")
print("  " + "─" * 58)
for i, (actual, pred, prob) in enumerate(
        zip(sample_y.values, sample_pred, sample_proba)):
    actual_lbl = "🟢 Legitimate" if actual == 0 else "🔴 FRAUD"
    pred_lbl   = "Legitimate"    if pred   == 0 else "⚠ FRAUD"
    status     = "✅ CORRECT" if actual == pred else "❌ WRONG"
    print(f"  {i+1:<4} {actual_lbl:<14} {pred_lbl:<14} {prob:>10.4f}  {status}")

# ================================================================
# 7. KEY INSIGHTS
# ================================================================
print("\n" + "=" * 65)
print("   KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 65)

print(f"""
🔑 FINDINGS:
   1. Class Imbalance   : {vc[0]/vc[1]:.0f}:1 ratio  — SMOTE/Undersampling essential
   2. Best Strategy     : {best_strat}
   3. Best Algorithm    : {best_model}
   4. Top Feature       : {importances.index[-1]} (most discriminative)

📊 WHY RECALL MATTERS MORE THAN ACCURACY:
   • A naive model predicting ALL as legit = {vc[0]/len(df)*100:.1f}% accuracy
   • But it catches ZERO fraud — useless!
   • We optimize for F1-Score & Recall to minimize missed frauds

⚠ TRADEOFF:
   • High Recall  → fewer frauds slip through (more false alarms)
   • High Precision → fewer false alarms (some fraud slips through)
   • F1-Score balances both — ideal metric for imbalanced problems

🛡 REAL-WORLD RECOMMENDATION:
   • Use {best_strat} + {best_model}
   • Set decision threshold at 0.3–0.4 (not default 0.5)
     to prioritize catching fraud over false alarms
""")

print("=" * 65)
print("   PROJECT COMPLETE  ✅")
print("=" * 65)
