# ================================================================
#   MOVIE RATING PREDICTION USING MACHINE LEARNING IN PYTHON
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

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Global Style ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d0d16',
    'axes.facecolor':   '#13131f',
    'axes.edgecolor':   '#333355',
    'axes.labelcolor':  '#dde0ff',
    'xtick.color':      '#9999bb',
    'ytick.color':      '#9999bb',
    'text.color':       '#dde0ff',
    'grid.color':       '#22223a',
    'grid.alpha':       0.6,
    'font.family':      'monospace',
})

GOLD   = '#f5c518'   # IMDb gold
TEAL   = '#00c9a7'
ROSE   = '#ff6b9d'
LAVR   = '#a78bfa'
BLUE   = '#60a5fa'
ORANGE = '#fb923c'
BG     = '#0d0d16'

# ================================================================
# 1. LOAD & EXPLORE DATA
# ================================================================
print("=" * 65)
print("   MOVIE RATING PREDICTION  —  DATA SCIENCE PROJECT")
print("=" * 65)

df = pd.read_csv('/home/claude/movies.csv')

print(f"\n📊 Dataset Shape   : {df.shape}")
print(f"📋 Columns         : {list(df.columns)}")
print(f"\n🔍 Missing Values  :\n{df.isnull().sum()}")
print(f"\n📈 Rating Stats    :")
print(df['Rating'].describe().round(3))
print(f"\n🎬 Unique Genres   : {df['Genre'].nunique()}  → {sorted(df['Genre'].unique())}")
print(f"🎬 Unique Directors: {df['Director'].nunique()}")
print(f"\n📋 Sample Records  :")
print(df[['Name','Year','Genre','Rating','Director']].head(8).to_string(index=False))

# ================================================================
# 2. PREPROCESSING & FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 65)
print("   FEATURE ENGINEERING")
print("=" * 65)

df = df.drop_duplicates().copy()
df.dropna(subset=['Rating'], inplace=True)

# ── Encode categoricals ─────────────────────────────────────
le_genre    = LabelEncoder()
le_director = LabelEncoder()
le_a1       = LabelEncoder()
le_a2       = LabelEncoder()
le_a3       = LabelEncoder()

df['Genre_enc']    = le_genre.fit_transform(df['Genre'])
df['Director_enc'] = le_director.fit_transform(df['Director'])
df['Actor1_enc']   = le_a1.fit_transform(df['Actor1'])
df['Actor2_enc']   = le_a2.fit_transform(df['Actor2'])
df['Actor3_enc']   = le_a3.fit_transform(df['Actor3'])

# ── Derived features ────────────────────────────────────────
df['Log_Votes']    = np.log1p(df['Votes'])
df['Movie_Age']    = 2024 - df['Year']
df['Duration_sq']  = df['Duration'] ** 2

# Director average rating (popularity proxy)
dir_avg = df.groupby('Director')['Rating'].transform('mean')
df['Director_avg_rating'] = dir_avg

# Genre average rating
genre_avg = df.groupby('Genre')['Rating'].transform('mean')
df['Genre_avg_rating'] = genre_avg

# Votes popularity tier
df['Popularity_tier'] = pd.cut(df['Votes'],
    bins=[0, 300000, 700000, 1200000, np.inf],
    labels=[0, 1, 2, 3]).astype(int)

print("\n✅ Features created:")
feature_cols = [
    'Duration', 'Genre_enc', 'Director_enc',
    'Actor1_enc', 'Actor2_enc', 'Actor3_enc',
    'Log_Votes', 'Movie_Age', 'Director_avg_rating',
    'Genre_avg_rating', 'Popularity_tier'
]
for f in feature_cols:
    print(f"   • {f}")

# ================================================================
# 3. EDA VISUALIZATION
# ================================================================

fig = plt.figure(figsize=(20, 16), facecolor=BG)
fig.suptitle('🎬  MOVIE RATING PREDICTION  —  EXPLORATORY DATA ANALYSIS',
             fontsize=17, color=GOLD, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

# Panel 1: Rating Distribution
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['Rating'], bins=20, color=GOLD, alpha=0.85,
         edgecolor=BG, linewidth=0.6, zorder=3)
ax1.axvline(df['Rating'].mean(), color=ROSE, linewidth=2.5,
            linestyle='--', label=f'Mean={df["Rating"].mean():.2f}')
ax1.axvline(df['Rating'].median(), color=TEAL, linewidth=2.5,
            linestyle=':', label=f'Median={df["Rating"].median():.2f}')
ax1.set_title('Rating Distribution', color=GOLD, fontsize=12, pad=8)
ax1.set_xlabel('IMDb Rating');  ax1.set_ylabel('Count')
ax1.legend(fontsize=9);  ax1.grid(True, alpha=0.35)

# Panel 2: Genre vs Rating
ax2 = fig.add_subplot(gs[0, 2:])
genre_stats = df.groupby('Genre')['Rating'].mean().sort_values(ascending=True)
colors_g = plt.cm.plasma(np.linspace(0.2, 0.9, len(genre_stats)))
bars = ax2.barh(genre_stats.index, genre_stats.values,
                color=colors_g, alpha=0.9, edgecolor=BG, linewidth=0.4)
for bar, val in zip(bars, genre_stats.values):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=8, color='white')
ax2.set_title('Average Rating by Genre', color=LAVR, fontsize=12, pad=8)
ax2.set_xlabel('Avg Rating');  ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, 10)

# Panel 3: Votes vs Rating scatter
ax3 = fig.add_subplot(gs[1, :2])
sc = ax3.scatter(df['Log_Votes'], df['Rating'],
                 c=df['Rating'], cmap='plasma', alpha=0.75,
                 edgecolors='white', linewidths=0.3, s=55, zorder=3)
m, b = np.polyfit(df['Log_Votes'], df['Rating'], 1)
x_line = np.linspace(df['Log_Votes'].min(), df['Log_Votes'].max(), 100)
ax3.plot(x_line, m*x_line+b, color=TEAL, linewidth=2, linestyle='--', label='Trend')
plt.colorbar(sc, ax=ax3, shrink=0.8, label='Rating')
ax3.set_title('Log(Votes) vs Rating', color=TEAL, fontsize=12, pad=8)
ax3.set_xlabel('Log(Votes)');  ax3.set_ylabel('Rating')
ax3.legend(fontsize=8);  ax3.grid(True, alpha=0.3)

# Panel 4: Duration vs Rating
ax4 = fig.add_subplot(gs[1, 2:])
ax4.scatter(df['Duration'], df['Rating'],
            c=df['Genre_enc'], cmap='tab10', alpha=0.75,
            edgecolors='white', linewidths=0.3, s=55)
m2, b2 = np.polyfit(df['Duration'], df['Rating'], 1)
x2 = np.linspace(df['Duration'].min(), df['Duration'].max(), 100)
ax4.plot(x2, m2*x2+b2, color=ROSE, linewidth=2.5, linestyle='--', label='Trend')
ax4.set_title('Duration vs Rating', color=ROSE, fontsize=12, pad=8)
ax4.set_xlabel('Duration (min)');  ax4.set_ylabel('Rating')
ax4.legend(fontsize=8);  ax4.grid(True, alpha=0.3)

# Panel 5: Year vs Rating
ax5 = fig.add_subplot(gs[2, :2])
ax5.scatter(df['Year'], df['Rating'],
            c=df['Rating'], cmap='plasma', alpha=0.75,
            edgecolors='white', linewidths=0.3, s=55)
# Decade average line
df['Decade'] = (df['Year'] // 10) * 10
decade_avg   = df.groupby('Decade')['Rating'].mean()
ax5.plot(decade_avg.index + 5, decade_avg.values,
         color=GOLD, linewidth=3, marker='D', markersize=6,
         markerfacecolor=GOLD, label='Decade Avg')
ax5.set_title('Release Year vs Rating', color=ORANGE, fontsize=12, pad=8)
ax5.set_xlabel('Year');  ax5.set_ylabel('Rating')
ax5.legend(fontsize=8);  ax5.grid(True, alpha=0.3)

# Panel 6: Correlation Heatmap
ax6 = fig.add_subplot(gs[2, 2:])
corr_cols = ['Rating','Log_Votes','Duration','Movie_Age',
             'Director_avg_rating','Genre_avg_rating','Popularity_tier']
corr_mat = df[corr_cols].corr()
sns.heatmap(corr_mat, ax=ax6, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0,
            linewidths=0.5, linecolor=BG,
            annot_kws={'size': 7, 'color': 'black'},
            cbar_kws={'shrink': 0.7})
ax6.set_title('Correlation Heatmap', color=BLUE, fontsize=12, pad=8)
ax6.tick_params(labelsize=7, rotation=30)

plt.savefig('/home/claude/eda_movies.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\n✅  EDA chart saved → eda_movies.png")

# ================================================================
# 4. MODEL TRAINING — MULTIPLE ALGORITHMS
# ================================================================
print("\n" + "=" * 65)
print("   MODEL TRAINING & COMPARISON")
print("=" * 65)

X = df[feature_cols]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Define Models ──────────────────────────────────────────
models = {
    'Linear Regression'        : LinearRegression(),
    'Ridge Regression'         : Ridge(alpha=1.0),
    'Random Forest'            : RandomForestRegressor(
                                    n_estimators=200, max_depth=6,
                                    random_state=42, n_jobs=-1),
    'Gradient Boosting'        : GradientBoostingRegressor(
                                    n_estimators=200, max_depth=4,
                                    learning_rate=0.08, random_state=42),
}

results    = {}
y_preds    = {}
cv_results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<28} {'MAE':>7} {'RMSE':>7} {'R²':>7}  {'CV_R²':>10}")
print("─" * 65)

for name, mdl in models.items():
    if name in ['Linear Regression', 'Ridge Regression']:
        mdl.fit(X_train_sc, y_train)
        y_pred = mdl.predict(X_test_sc)
        cv     = cross_val_score(mdl, scaler.transform(X), y,
                                 cv=kf, scoring='r2')
    else:
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        cv     = cross_val_score(mdl, X, y, cv=kf, scoring='r2')

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name]    = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    y_preds[name]    = y_pred
    cv_results[name] = cv

    print(f"  {name:<26} {mae:>7.4f} {rmse:>7.4f} {r2:>7.4f}  "
          f"{cv.mean():.4f}±{cv.std():.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['R2'])
best_pred = y_preds[best_name]
best_mdl  = models[best_name]
print(f"\n🏆 Best Model : {best_name}  (R² = {results[best_name]['R2']:.4f})")

# Feature importance (from Random Forest)
rf_model   = models['Random Forest']
feat_imp   = pd.Series(rf_model.feature_importances_, index=feature_cols)
feat_imp   = feat_imp.sort_values(ascending=True)

# ================================================================
# 5. RESULTS VISUALIZATION
# ================================================================

fig2 = plt.figure(figsize=(20, 16), facecolor=BG)
fig2.suptitle('🎬  MOVIE RATING PREDICTION  —  MODEL RESULTS',
              fontsize=17, color=GOLD, fontweight='bold', y=0.98)
gs2 = gridspec.GridSpec(3, 3, figure=fig2, hspace=0.52, wspace=0.38)

model_colors = [TEAL, LAVR, GOLD, ROSE]
m_names      = list(results.keys())

# Panel 1: Actual vs Predicted — best model
ax1 = fig2.add_subplot(gs2[0, :2])
ax1.scatter(y_test, best_pred,
            color=GOLD, alpha=0.8, edgecolors='white',
            linewidths=0.4, s=70, zorder=4, label='Predictions')
lims = [min(y_test.min(), best_pred.min())-0.3,
        max(y_test.max(), best_pred.max())+0.3]
ax1.plot(lims, lims, '--', color=TEAL, linewidth=2.5,
         label='Perfect Fit', zorder=3)
ax1.fill_between(lims, [l-0.5 for l in lims], [l+0.5 for l in lims],
                 alpha=0.08, color=TEAL, label='±0.5 band')
ax1.set_xlim(lims);  ax1.set_ylim(lims)
ax1.set_xlabel('Actual Rating');  ax1.set_ylabel('Predicted Rating')
ax1.set_title(f'Actual vs Predicted  [{best_name}]',
              color=GOLD, fontsize=12, pad=8)
ax1.legend(fontsize=9);  ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.90, f'R² = {results[best_name]["R2"]:.4f}',
         transform=ax1.transAxes, fontsize=11, color=GOLD,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor=GOLD))

# Panel 2: R² comparison bar
ax2 = fig2.add_subplot(gs2[0, 2])
r2_vals = [results[n]['R2'] for n in m_names]
bars2   = ax2.barh(m_names, r2_vals, color=model_colors,
                   alpha=0.9, edgecolor=BG, linewidth=0.4)
for bar, val in zip(bars2, r2_vals):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=9, color='white')
ax2.set_title('R² Score Comparison', color=BLUE, fontsize=11, pad=8)
ax2.set_xlabel('R² Score');  ax2.set_xlim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='x')
ax2.tick_params(labelsize=8)

# Panel 3: Residuals
ax3 = fig2.add_subplot(gs2[1, :2])
residuals = y_test.values - best_pred
ax3.scatter(best_pred, residuals, color=ORANGE, alpha=0.8,
            edgecolors='white', linewidths=0.3, s=55)
ax3.axhline(0, color=TEAL, linewidth=2.5, linestyle='--')
ax3.axhline(residuals.std(), color=ROSE, linewidth=1.2,
            linestyle=':', label=f'+1σ ({residuals.std():.3f})')
ax3.axhline(-residuals.std(), color=ROSE, linewidth=1.2,
            linestyle=':', label=f'-1σ')
ax3.set_xlabel('Predicted Rating');  ax3.set_ylabel('Residual')
ax3.set_title('Residual Plot', color=ORANGE, fontsize=12, pad=8)
ax3.legend(fontsize=8);  ax3.grid(True, alpha=0.3)

# Panel 4: Feature Importance
ax4 = fig2.add_subplot(gs2[1, 2])
feat_clrs = plt.cm.viridis(np.linspace(0.2, 0.9, len(feat_imp)))
h_bars = ax4.barh(feat_imp.index, feat_imp.values,
                  color=feat_clrs, alpha=0.9,
                  edgecolor=BG, linewidth=0.4)
for bar, val in zip(h_bars, feat_imp.values):
    ax4.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=7.5, color='white')
ax4.set_title('Feature Importance\n(Random Forest)', color=TEAL, fontsize=11, pad=8)
ax4.set_xlabel('Importance Score')
ax4.grid(True, alpha=0.3, axis='x')
ax4.tick_params(labelsize=8)

# Panel 5: CV Scores box
ax5 = fig2.add_subplot(gs2[2, :2])
cv_data = [cv_results[n] for n in m_names]
bp = ax5.boxplot(cv_data, labels=[n.replace(' ', '\n') for n in m_names],
                 patch_artist=True, notch=False,
                 medianprops=dict(color='white', linewidth=2.5))
for patch, clr in zip(bp['boxes'], model_colors):
    patch.set_facecolor(clr);  patch.set_alpha(0.75)
for element in ['whiskers', 'caps']:
    for item in bp[element]:
        item.set_color('#666688')
for flier in bp['fliers']:
    flier.set_markerfacecolor('#aaaacc');  flier.set_markersize(4)
ax5.set_title('Cross-Validation R² Scores (5-Fold)', color=ROSE, fontsize=12, pad=8)
ax5.set_ylabel('R²');  ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: Metrics Dashboard
ax6 = fig2.add_subplot(gs2[2, 2])
ax6.axis('off')
metrics = [
    ('R²  Score',  f'{results[best_name]["R2"]:.4f}',   GOLD),
    ('MAE',        f'{results[best_name]["MAE"]:.4f}',   TEAL),
    ('RMSE',       f'{results[best_name]["RMSE"]:.4f}',  ROSE),
    ('CV R² Mean', f'{cv_results[best_name].mean():.4f}', LAVR),
]
ax6.text(0.5, 0.97, f'Best Model:', ha='center', va='top',
         transform=ax6.transAxes, fontsize=9, color='#888888')
ax6.text(0.5, 0.90, best_name, ha='center', va='top',
         transform=ax6.transAxes, fontsize=10, color=GOLD, fontweight='bold')
for i, (label, val, clr) in enumerate(metrics):
    yp = 0.72 - i * 0.185
    rect = plt.Rectangle((0.05, yp-0.07), 0.9, 0.14,
                          fill=True, facecolor='#1a1a2e',
                          edgecolor=clr, linewidth=1.5,
                          transform=ax6.transAxes)
    ax6.add_patch(rect)
    ax6.text(0.5, yp + 0.03, val, ha='center', va='center',
             transform=ax6.transAxes, fontsize=15,
             color=clr, fontweight='bold')
    ax6.text(0.5, yp - 0.04, label, ha='center', va='center',
             transform=ax6.transAxes, fontsize=8, color='#9999bb')
ax6.set_title('Performance Dashboard', color=LAVR, fontsize=11, pad=8)

plt.savefig('/home/claude/model_results_movies.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("✅  Model results chart saved → model_results_movies.png")

# ================================================================
# 6. SAMPLE PREDICTIONS
# ================================================================
print("\n" + "=" * 65)
print("   SAMPLE PREDICTIONS  (Random Forest Model)")
print("=" * 65)

def predict_rating(name, year, duration, genre, director, votes,
                   actor1='Unknown', actor2='Unknown', actor3='Unknown'):
    genre_e = le_genre.transform([genre])[0] if genre in le_genre.classes_ \
              else le_genre.transform([le_genre.classes_[0]])[0]
    dir_e   = le_director.transform([director])[0] if director in le_director.classes_ \
              else le_director.transform([le_director.classes_[0]])[0]
    a1_e    = le_a1.transform([actor1])[0] if actor1 in le_a1.classes_ \
              else le_a1.transform([le_a1.classes_[0]])[0]
    a2_e    = le_a2.transform([actor2])[0] if actor2 in le_a2.classes_ \
              else le_a2.transform([le_a2.classes_[0]])[0]
    a3_e    = le_a3.transform([actor3])[0] if actor3 in le_a3.classes_ \
              else le_a3.transform([le_a3.classes_[0]])[0]

    log_v   = np.log1p(votes)
    age     = 2024 - year
    dir_avg_r  = df[df['Director_enc']==dir_e]['Director_avg_rating'].mean()
    dir_avg_r  = dir_avg_r if not np.isnan(dir_avg_r) else df['Director_avg_rating'].mean()
    gen_avg_r  = df[df['Genre_enc']==genre_e]['Genre_avg_rating'].mean()
    gen_avg_r  = gen_avg_r if not np.isnan(gen_avg_r) else df['Genre_avg_rating'].mean()
    pop_tier   = int(pd.cut([votes], bins=[0,300000,700000,1200000,np.inf],
                            labels=[0,1,2,3])[0])

    row = np.array([[duration, genre_e, dir_e, a1_e, a2_e, a3_e,
                     log_v, age, dir_avg_r, gen_avg_r, pop_tier]])
    pred = rf_model.predict(row)[0]
    return round(float(np.clip(pred, 1, 10)), 2)

test_movies = [
    ('Oppenheimer',  2023, 180, 'Drama',    'Christopher Nolan', 1100000),
    ('Barbie',       2023, 114, 'Comedy',   'Greta Gerwig',       900000),
    ('John Wick 4',  2023, 169, 'Action',   'Chad Stahelski',     600000),
    ('Low Budget Film', 2020, 90, 'Horror', 'Unknown Director',    50000),
]

print(f"\n  {'Movie':<25} {'Genre':<12} {'Votes':>8}  → Predicted Rating")
print("  " + "─"*60)
for (nm, yr, dur, genre, director, votes) in test_movies:
    p = predict_rating(nm, yr, dur, genre, director, votes)
    bar = '█' * int(p) + '░' * (10 - int(p))
    print(f"  {nm:<25} {genre:<12} {votes:>8,}  →  ⭐ {p}  {bar}")

# ================================================================
# 7. KEY INSIGHTS
# ================================================================
print("\n" + "=" * 65)
print("   KEY INSIGHTS FROM ANALYSIS")
print("=" * 65)
top_feat = feat_imp.sort_values(ascending=False)
print("\n🔑 Top Factors Influencing Movie Ratings:")
for i, (feat, imp) in enumerate(top_feat.items(), 1):
    print(f"   {i}. {feat:<30} → {imp:.4f} importance")

genre_best = df.groupby('Genre')['Rating'].mean().idxmax()
genre_low  = df.groupby('Genre')['Rating'].mean().idxmin()
print(f"\n🎭 Highest Rated Genre : {genre_best}")
print(f"🎭 Lowest  Rated Genre : {genre_low}")
print(f"\n📊 Correlation(Votes, Rating) : {df['Rating'].corr(df['Log_Votes']):.4f}")
print(f"📊 Correlation(Year,  Rating) : {df['Rating'].corr(df['Year']):.4f}")

print("\n" + "=" * 65)
print("   PROJECT COMPLETE  ✅")
print("=" * 65)
