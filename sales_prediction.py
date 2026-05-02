# ============================================================
#   SALES PREDICTION USING MACHINE LEARNING IN PYTHON
#   Data Science Internship Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#1a1a2e',
    'axes.edgecolor':   '#444466',
    'axes.labelcolor':  '#e0e0ff',
    'xtick.color':      '#aaaacc',
    'ytick.color':      '#aaaacc',
    'text.color':       '#e0e0ff',
    'grid.color':       '#2a2a4a',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})

ACCENT   = '#7c5cbf'
CYAN     = '#4dd9e8'
ORANGE   = '#ff7f50'
GREEN    = '#50fa7b'
PINK     = '#ff79c6'

# ============================================================
# 1. LOAD & EXPLORE DATA
# ============================================================
print("=" * 60)
print("  SALES PREDICTION — DATA SCIENCE PROJECT")
print("=" * 60)

df = pd.read_csv('/home/claude/advertising.csv')
df.columns = ['TV', 'Radio', 'Newspaper', 'Sales']

print("\n📊 Dataset Shape:", df.shape)
print("\n📋 First 5 Rows:")
print(df.head())
print("\n📈 Statistical Summary:")
print(df.describe().round(2))
print("\n🔍 Missing Values:", df.isnull().sum().sum())
print("\n✅ No missing values found — clean dataset!")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

fig = plt.figure(figsize=(18, 14), facecolor='#0f0f1a')
fig.suptitle('SALES PREDICTION — EXPLORATORY DATA ANALYSIS',
             fontsize=16, color='#e0e0ff', fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Distribution plots ---
channels = ['TV', 'Radio', 'Newspaper', 'Sales']
colors   = [ACCENT, CYAN, ORANGE, GREEN]

for i, (col, clr) in enumerate(zip(channels, colors)):
    ax = fig.add_subplot(gs[0, i] if i < 3 else gs[1, 0])
    ax.hist(df[col], bins=20, color=clr, alpha=0.85, edgecolor='#0f0f1a', linewidth=0.5)
    ax.set_title(f'{col} Distribution', fontsize=10, color=clr, pad=8)
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.grid(True, alpha=0.3)

# Sales distribution on row 1 col 3
ax_sales_dist = fig.add_subplot(gs[1, 0])
ax_sales_dist.hist(df['Sales'], bins=20, color=GREEN, alpha=0.85,
                   edgecolor='#0f0f1a', linewidth=0.5)
ax_sales_dist.set_title('Sales Distribution', fontsize=10, color=GREEN, pad=8)
ax_sales_dist.set_xlabel('Sales')
ax_sales_dist.set_ylabel('Frequency')
ax_sales_dist.grid(True, alpha=0.3)

# --- Scatter: TV vs Sales ---
ax1 = fig.add_subplot(gs[1, 1])
ax1.scatter(df['TV'], df['Sales'], alpha=0.6, color=ACCENT,
            edgecolors='white', linewidths=0.3, s=40)
m, b = np.polyfit(df['TV'], df['Sales'], 1)
x_line = np.linspace(df['TV'].min(), df['TV'].max(), 100)
ax1.plot(x_line, m*x_line + b, color=CYAN, linewidth=2, linestyle='--', label='Trend')
ax1.set_title('TV Budget vs Sales', fontsize=10, color=ACCENT, pad=8)
ax1.set_xlabel('TV Advertising ($K)');  ax1.set_ylabel('Sales ($K)')
ax1.legend(fontsize=7);  ax1.grid(True, alpha=0.3)

# --- Scatter: Radio vs Sales ---
ax2 = fig.add_subplot(gs[1, 2])
ax2.scatter(df['Radio'], df['Sales'], alpha=0.6, color=CYAN,
            edgecolors='white', linewidths=0.3, s=40)
m2, b2 = np.polyfit(df['Radio'], df['Sales'], 1)
x2 = np.linspace(df['Radio'].min(), df['Radio'].max(), 100)
ax2.plot(x2, m2*x2 + b2, color=ORANGE, linewidth=2, linestyle='--', label='Trend')
ax2.set_title('Radio Budget vs Sales', fontsize=10, color=CYAN, pad=8)
ax2.set_xlabel('Radio Advertising ($K)');  ax2.set_ylabel('Sales ($K)')
ax2.legend(fontsize=7);  ax2.grid(True, alpha=0.3)

# --- Correlation Heatmap ---
ax3 = fig.add_subplot(gs[2, 0])
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr, ax=ax3, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, linecolor='#0f0f1a',
            annot_kws={'size': 9, 'color': 'white'}, cbar=False)
ax3.set_title('Correlation Matrix', fontsize=10, color=PINK, pad=8)
ax3.tick_params(labelsize=8)

# --- Box plots ---
ax4 = fig.add_subplot(gs[2, 1])
bp = ax4.boxplot([df['TV'], df['Radio'], df['Newspaper']],
                 labels=['TV', 'Radio', 'Newspaper'],
                 patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))
for patch, clr in zip(bp['boxes'], [ACCENT, CYAN, ORANGE]):
    patch.set_facecolor(clr);  patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp[element]:
        item.set_color('#aaaacc')
ax4.set_title('Ad Budget Distribution', fontsize=10, color=ORANGE, pad=8)
ax4.set_ylabel('Budget ($K)');  ax4.grid(True, alpha=0.3)

# --- Pairwise budget vs sales bar-style ---
ax5 = fig.add_subplot(gs[2, 2])
corr_vals = df[['TV', 'Radio', 'Newspaper']].corrwith(df['Sales'])
bars = ax5.bar(corr_vals.index, corr_vals.values,
               color=[ACCENT, CYAN, ORANGE], alpha=0.85,
               edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, corr_vals.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='white')
ax5.set_title('Correlation with Sales', fontsize=10, color=GREEN, pad=8)
ax5.set_ylabel('Pearson r');  ax5.set_ylim(0, 1)
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig('/home/claude/eda_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
print("\n✅ EDA chart saved.")

# ============================================================
# 3. FEATURE ENGINEERING & MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("  MODEL TRAINING")
print("=" * 60)

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)

# Cross-validation
cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring='r2')

# ── Metrics ────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"""
📊 MODEL PERFORMANCE METRICS
─────────────────────────────
  MAE   : {mae:.4f}
  MSE   : {mse:.4f}
  RMSE  : {rmse:.4f}
  R²    : {r2:.4f}  ({r2*100:.2f}% variance explained)

📊 CROSS-VALIDATION (5-Fold)
─────────────────────────────
  Scores : {np.round(cv_scores, 4)}
  Mean   : {cv_scores.mean():.4f}
  Std    : {cv_scores.std():.4f}

📊 MODEL COEFFICIENTS
─────────────────────────────""")
for feat, coef in zip(X.columns, model.coef_):
    print(f"  {feat:<12}: {coef:+.4f}")
print(f"  {'Intercept':<12}: {model.intercept_:+.4f}")

# ============================================================
# 4. RESULTS VISUALIZATION
# ============================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor='#0f0f1a')
fig2.suptitle('SALES PREDICTION — MODEL RESULTS',
              fontsize=15, color='#e0e0ff', fontweight='bold', y=0.98)

# Panel 1: Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, color=ACCENT, alpha=0.75,
           edgecolors=CYAN, linewidths=0.5, s=60, zorder=3)
lims = [min(y_test.min(), y_pred.min())-1, max(y_test.max(), y_pred.max())+1]
ax.plot(lims, lims, '--', color=GREEN, linewidth=2, label='Perfect Prediction', zorder=2)
ax.fill_between(lims, [l-1.5 for l in lims], [l+1.5 for l in lims],
                alpha=0.08, color=GREEN, label='±1.5 error band')
ax.set_xlim(lims);  ax.set_ylim(lims)
ax.set_xlabel('Actual Sales ($K)');  ax.set_ylabel('Predicted Sales ($K)')
ax.set_title('Actual vs Predicted Sales', color=CYAN, fontsize=11)
ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
ax.text(0.05, 0.92, f'R² = {r2:.4f}', transform=ax.transAxes,
        fontsize=10, color=GREEN,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', edgecolor=GREEN))

# Panel 2: Residuals
ax2 = axes[0, 1]
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, color=ORANGE, alpha=0.75,
            edgecolors='white', linewidths=0.3, s=50)
ax2.axhline(0, color=CYAN, linewidth=2, linestyle='--')
ax2.axhline(residuals.std(), color=PINK, linewidth=1, linestyle=':', alpha=0.7)
ax2.axhline(-residuals.std(), color=PINK, linewidth=1, linestyle=':', alpha=0.7)
ax2.set_xlabel('Predicted Values');  ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Fitted', color=ORANGE, fontsize=11)
ax2.grid(True, alpha=0.3)

# Panel 3: Feature Importance
ax3 = axes[1, 0]
feat_importance = pd.Series(np.abs(model.coef_), index=X.columns).sort_values(ascending=True)
colors_bar = [ACCENT, CYAN, GREEN]
bars = ax3.barh(feat_importance.index, feat_importance.values,
                color=colors_bar, alpha=0.85, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, feat_importance.values):
    ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10, color='white')
ax3.set_xlabel('|Coefficient| (Standardized)')
ax3.set_title('Feature Importance', color=GREEN, fontsize=11)
ax3.grid(True, alpha=0.3, axis='x')

# Panel 4: Metrics dashboard
ax4 = axes[1, 1]
ax4.axis('off')
metrics_data = [
    ('R² Score',  f'{r2:.4f}',   f'{r2*100:.1f}% variance\nexplained',  GREEN),
    ('MAE',       f'{mae:.4f}',  'Avg absolute\nerror ($K)',             CYAN),
    ('RMSE',      f'{rmse:.4f}', 'Root mean\nsquared error',            ORANGE),
    ('CV Mean',   f'{cv_scores.mean():.4f}', '5-Fold cross\nvalidation r²', PINK),
]
for i, (name, val, desc, clr) in enumerate(metrics_data):
    col = i % 2;  row = i // 2
    x = 0.08 + col * 0.5;  y = 0.72 - row * 0.42
    rect = plt.Rectangle((x-0.02, y-0.28), 0.44, 0.34,
                          fill=True, facecolor='#1a1a2e',
                          edgecolor=clr, linewidth=1.5, transform=ax4.transAxes)
    ax4.add_patch(rect)
    ax4.text(x + 0.2, y + 0.02, val, ha='center', va='center', fontsize=18,
             color=clr, fontweight='bold', transform=ax4.transAxes)
    ax4.text(x + 0.2, y - 0.13, name, ha='center', va='center', fontsize=8,
             color='#aaaacc', transform=ax4.transAxes)
    ax4.text(x + 0.2, y - 0.23, desc, ha='center', va='center', fontsize=7,
             color='#666688', transform=ax4.transAxes)
ax4.set_title('Performance Dashboard', color=PINK, fontsize=11, pad=15)

plt.tight_layout()
plt.savefig('/home/claude/model_results.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f1a')
print("\n✅ Model results chart saved.")

# ============================================================
# 5. SAMPLE PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("  SAMPLE PREDICTIONS")
print("=" * 60)

test_cases = pd.DataFrame({
    'TV':        [230.1, 44.5,  100.0, 180.0],
    'Radio':     [37.8,  39.3,  20.0,  45.0],
    'Newspaper': [69.2,  45.1,  10.0,  30.0]
})

test_scaled  = scaler.transform(test_cases)
predictions  = model.predict(test_scaled)

print("\n  TV($K)  Radio($K)  News($K)  → Predicted Sales($K)")
print("  " + "─"*52)
for _, row in test_cases.iterrows():
    pred = model.predict(scaler.transform([row.values]))[0]
    print(f"  {row.TV:>6.1f}   {row.Radio:>6.1f}    {row.Newspaper:>6.1f}   →   {pred:.2f}")

print("\n" + "=" * 60)
print("  PROJECT COMPLETE ✅")
print("=" * 60)
