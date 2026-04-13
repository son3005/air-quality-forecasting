"""
╔══════════════════════════════════════════════════════════════════════════╗
║  BENCHMARK ANALYSIS — Air Quality Forecasting PM2.5                     ║
║  Phân tích toàn diện hiệu suất 6 mô hình × 5 tầm nhìn × 3 block       ║
║  Input:  report/benchmark_metrics.json                                  ║
║  Output: report/figures/*.png                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════
# CELL 1: Import & Setup
# ══════════════════════════════════════════════════════════════════════════
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'legend.fontsize': 9,
})

# Color palette cho 6 models
MODEL_COLORS = {
    'XGBoost':      '#f0883e',
    'XLinear':      '#58a6ff',
    'iTransformer': '#bc8cff',
    'ESTGCN':       '#3fb950',
    'ST-XLinear':   '#f778ba',
    'Ensemble':     '#ff7b72',
}
MODEL_MARKERS = {
    'XGBoost': 'o', 'XLinear': 's', 'iTransformer': '^',
    'ESTGCN': 'D', 'ST-XLinear': 'P', 'Ensemble': '*',
}

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(fig, name):
    fig.savefig(os.path.join(SAVE_DIR, f'{name}.png'), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  💾 Saved: figures/{name}.png")


# ══════════════════════════════════════════════════════════════════════════
# CELL 2: Load & Transform Data
# ══════════════════════════════════════════════════════════════════════════
with open(os.path.join(os.path.dirname(__file__), 'benchmark_metrics.json'), 'r', encoding='utf-8') as f:
    raw = json.load(f)

HORIZONS = raw['horizons']       # ['T+1', ..., 'T+24']
MODELS   = raw['models']         # 6 models
METRICS  = raw['metrics']        # ['RMSE', 'MAE', 'R2', 'MAPE']
BLOCKS   = list(raw['data'].keys())  # ['block5', 'block7', 'block30']
REGIONS  = list(raw['data'][BLOCKS[0]].keys())  # ['north', 'south']
HORIZON_NUMS = [1, 3, 6, 12, 24]

# Build flat DataFrame: (block, region, model, horizon, metric) → value
rows = []
for block in BLOCKS:
    for region in REGIONS:
        for metric in METRICS:
            for model in MODELS:
                vals = raw['data'][block][region][metric].get(model, [None]*5)
                for hi, h_label in enumerate(HORIZONS):
                    rows.append({
                        'block': block, 'region': region, 'model': model,
                        'horizon': h_label, 'horizon_h': HORIZON_NUMS[hi],
                        'metric': metric, 'value': float(vals[hi]) if vals[hi] is not None else np.nan
                    })
        # Training time
        if 'train_time' in raw['data'][block][region]:
            for model in MODELS:
                tt = raw['data'][block][region]['train_time'].get(model, [None]*5)
                for hi, h_label in enumerate(HORIZONS):
                    rows.append({
                        'block': block, 'region': region, 'model': model,
                        'horizon': h_label, 'horizon_h': HORIZON_NUMS[hi],
                        'metric': 'train_time', 'value': float(tt[hi]) if tt[hi] is not None else np.nan
                    })

df = pd.DataFrame(rows)

# Weighted average across regions (for summary)
def weighted_avg(block, metric):
    """Average north + south per model × horizon."""
    sub = df[(df['block'] == block) & (df['metric'] == metric)]
    return sub.groupby(['model', 'horizon_h'])['value'].mean().unstack()

print(f"✅ Loaded {len(df)} records: {len(BLOCKS)} blocks × {len(REGIONS)} regions × {len(MODELS)} models × {len(HORIZONS)} horizons")
print(f"   Blocks: {BLOCKS}")
print(f"   Models: {MODELS}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 3: FIGURE 1 — MAE vs Horizon (per block, avg regions)
# Ý nghĩa: Cho thấy mức suy giảm hiệu suất khi tầm nhìn dự báo tăng.
#           Model nào có đường cong thoải hơn → giữ chất lượng tốt hơn ở 
#           dự báo dài hạn. XGBoost thường dominate do feature engineering.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
fig.suptitle('Hình 1: MAE theo tầm nhìn dự báo — So sánh đa block', y=1.02)

for ax_i, block in enumerate(BLOCKS):
    ax = axes[ax_i]
    pivot = weighted_avg(block, 'MAE')
    for model in MODELS:
        if model in pivot.index:
            vals = pivot.loc[model]
            ax.plot(HORIZON_NUMS, vals, marker=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model], label=model, linewidth=2,
                    markersize=7, alpha=0.9)
    ax.set_title(block.replace('block', 'Block '), fontsize=12)
    ax.set_xlabel('Tầm nhìn dự báo (giờ)')
    ax.set_ylabel('MAE (µg/m³)' if ax_i == 0 else '')
    ax.set_xticks(HORIZON_NUMS)
    ax.grid(True, linestyle='--', alpha=0.3)
    if ax_i == 2:
        ax.legend(loc='upper left', framealpha=0.8)

fig.tight_layout()
savefig(fig, '01_mae_vs_horizon_all_blocks')
plt.close()

print("""
📊 Hình 1 — MAE theo tầm nhìn dự báo:
   • XGBoost dẫn đầu ổn định trên mọi tầm nhìn nhờ feature engineering thủ công (lag, rolling, KNN neighbor).
   • Nhóm Deep Learning (XLinear, iTransformer, ST-XLinear) có MAE thấp ở T+1 nhưng suy giảm nhanh ở T+12, T+24.
   • Ensemble không luôn tốt nhất — chỉ hiệu quả khi 2 thành phần bổ trợ lẫn nhau.
   • Block 7 cho MAE tổng thể thấp hơn Block 5/30, cho thấy chu kỳ 7 ngày phản ánh tốt nhất quy luật tuần.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 4: FIGURE 2 — R² Heatmap (block7, both regions)
# Ý nghĩa: Heatmap thể hiện khả năng giải thích phương sai của mô hình.
#           Ô màu đậm = model giải thích tốt biến thiên PM2.5.
#           So sánh North vs South cho thấy ảnh hưởng địa lý.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Hình 2: R² Heatmap — Block 7 (North vs South)', y=1.02)

for ax_i, region in enumerate(REGIONS):
    ax = axes[ax_i]
    sub = df[(df['block'] == 'block7') & (df['region'] == region) & (df['metric'] == 'R2')]
    pivot = sub.pivot_table(index='model', columns='horizon', values='value')
    pivot = pivot[HORIZONS]  # Ensure order
    pivot = pivot.loc[[m for m in MODELS if m in pivot.index]]
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f'{region.upper()}', fontsize=12)
    
    # Annotate
    for i in range(len(pivot.index)):
        for j in range(len(HORIZONS)):
            val = pivot.values[i, j]
            color = 'white' if val < 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
savefig(fig, '02_r2_heatmap_block7')
plt.close()

print("""
📊 Hình 2 — R² Heatmap:
   • North: R² giảm mạnh từ 0.79–0.81 (T+1) xuống 0.06–0.52 (T+24). XGBoost giữ R²=0.52 ở T+24, cao nhất.
   • South: R² cao hơn North ở tầm gần (0.95–0.99) do phân phối PM2.5 ổn định hơn, nhưng sụt nhanh ở T+24.
   • ESTGCN cho R² thấp nhất — cho thấy graph tĩnh chưa đủ nắm bắt biến thiên, đặc biệt ở miền Bắc.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 5: FIGURE 3 — Grouped Bar Chart: MAE at T+1 vs T+24 (all blocks)
# Ý nghĩa: Trực quan hóa khoảng cách hiệu suất ngắn hạn vs dài hạn.
#           Chênh lệch càng lớn → model càng kém ổn định ở dự báo xa.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Hình 3: MAE ở T+1 vs T+24 — Khoảng cách hiệu suất ngắn/dài hạn', y=1.02)

for ax_i, block in enumerate(BLOCKS):
    ax = axes[ax_i]
    pivot = weighted_avg(block, 'MAE')
    models_order = [m for m in MODELS if m in pivot.index]
    
    x = np.arange(len(models_order))
    width = 0.35
    
    vals_t1 = [pivot.loc[m, 1] if 1 in pivot.columns else 0 for m in models_order]
    vals_t24 = [pivot.loc[m, 24] if 24 in pivot.columns else 0 for m in models_order]
    
    bars1 = ax.bar(x - width/2, vals_t1, width, label='T+1', 
                   color=[MODEL_COLORS[m] for m in models_order], alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_t24, width, label='T+24',
                   color=[MODEL_COLORS[m] for m in models_order], alpha=0.45, 
                   edgecolor=[MODEL_COLORS[m] for m in models_order], linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('iTransformer', 'iTrans') for m in models_order], rotation=30, ha='right')
    ax.set_ylabel('MAE (µg/m³)' if ax_i == 0 else '')
    ax.set_title(block.replace('block', 'Block '))
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Annotate delta
    for i, m in enumerate(models_order):
        delta = vals_t24[i] - vals_t1[i]
        ax.annotate(f'+{delta:.1f}', xy=(i + width/2, vals_t24[i]), 
                    fontsize=8, ha='center', va='bottom', color='#ff7b72')

fig.tight_layout()
savefig(fig, '03_mae_t1_vs_t24_gap')
plt.close()

print("""
📊 Hình 3 — Khoảng cách MAE (T+1 → T+24):
   • XGBoost: delta nhỏ nhất (~+7–8 µg/m³), cho thấy feature engineering giúp duy trì ổn định dài hạn.
   • iTransformer & ESTGCN: delta lớn nhất (~+12-16), suy giảm mạnh nhất khi horizon mở rộng.
   • ST-XLinear: delta trung bình, nhỉnh hơn XLinear thuần nhờ thông tin spatial.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 6: FIGURE 4 — RMSE Box Plot across blocks (per model, T+6 horizon)
# Ý nghĩa: Đánh giá tính ổn định của mô hình qua các block split khác nhau.
#           Box hẹp = mô hình ổn định, không phụ thuộc cách chia dữ liệu.
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Hình 4: Phân bố RMSE tại T+6 qua 3 block split (North + South)', y=1.01)

box_data = []
labels = []
colors = []
for model in MODELS:
    vals = []
    for block in BLOCKS:
        for region in REGIONS:
            sub = df[(df['block'] == block) & (df['region'] == region) & 
                     (df['model'] == model) & (df['metric'] == 'RMSE') & 
                     (df['horizon_h'] == 6)]
            vals.extend(sub['value'].tolist())
    box_data.append(vals)
    labels.append(model.replace('iTransformer', 'iTrans'))
    colors.append(MODEL_COLORS[model])

bp = ax.boxplot(box_data, labels=labels, patch_artist=True, notch=True,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('RMSE (µg/m³)')
ax.set_xlabel('Mô hình')
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Add individual points
for i, (data, color) in enumerate(zip(box_data, colors)):
    x = np.random.normal(i+1, 0.04, size=len(data))
    ax.scatter(x, data, color=color, alpha=0.6, s=25, zorder=3, edgecolors='white', linewidths=0.5)

fig.tight_layout()
savefig(fig, '04_rmse_boxplot_t6')
plt.close()

print("""
📊 Hình 4 — Box plot RMSE tại T+6:
   • XGBoost: box hẹp nhất + median thấp nhất → ổn định qua mọi cách chia dữ liệu.
   • ESTGCN & iTransformer: spread lớn giữa North/South do ảnh hưởng phân phối dữ liệu địa lý.
   • North nhất quán cho RMSE thấp hơn South do nồng độ PM2.5 phía Nam cao và biến động mạnh hơn.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 7: FIGURE 5 — Radar Chart: T+1 performance (block7, normalized)
# Ý nghĩa: So sánh đa chiều 4 metric cùng lúc ở tầm nhìn ngắn nhất.
#           Model càng phủ rộng radar → càng cân bằng trên mọi tiêu chí.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
fig.suptitle('Hình 5: Radar Chart — Hiệu suất T+1 đa metric (Block 7)', y=1.05)

radar_metrics = ['MAE', 'RMSE', 'MAPE']  # Lower is better
# For R2 we want higher = better, so invert for "lower = better" display
# Actually let's show all as "score" where higher = better

for ax_i, region in enumerate(REGIONS):
    ax = axes[ax_i]
    categories = ['1/MAE', '1/RMSE', 'R²', '1/MAPE']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for model in MODELS:
        sub = df[(df['block'] == 'block7') & (df['region'] == region) & 
                 (df['model'] == model) & (df['horizon_h'] == 1)]
        
        mae_val = sub[sub['metric'] == 'MAE']['value'].values
        rmse_val = sub[sub['metric'] == 'RMSE']['value'].values
        r2_val = sub[sub['metric'] == 'R2']['value'].values
        mape_val = sub[sub['metric'] == 'MAPE']['value'].values
        
        if len(mae_val) == 0: continue
        
        # Normalize to [0, 1] score where 1 = best
        scores = [
            1.0 / (mae_val[0] + 1),
            1.0 / (rmse_val[0] + 1),
            r2_val[0],
            1.0 / (mape_val[0] + 1),
        ]
        # Scale for visibility
        scores = [s * 10 for s in scores]
        scores += scores[:1]
        
        ax.plot(angles, scores, color=MODEL_COLORS[model], linewidth=2, label=model, alpha=0.8)
        ax.fill(angles, scores, color=MODEL_COLORS[model], alpha=0.05)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f'{region.upper()}', fontsize=12, pad=20)
    if ax_i == 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), framealpha=0.8)

fig.tight_layout()
savefig(fig, '05_radar_t1_block7')
plt.close()

print("""
📊 Hình 5 — Radar Chart T+1:
   • XGBoost: diện tích radar lớn nhất → cân bằng tốt nhất trên cả 4 metric đồng thời.
   • ESTGCN: profile lệch do MAE/RMSE cao nhưng R² vẫn chấp nhận → model chịu bias lớn ở giá trị cực đoan.
   • North vs South: South có R² cao hơn (phân phối ổn định) nhưng RMSE/MAE cao hơn (giá trị tuyệt đối PM2.5 lớn hơn).
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 8: FIGURE 6 — Training Time Comparison (stacked bar)
# Ý nghĩa: Chi phí tính toán thực tế — yếu tố quan trọng khi triển khai.
#           XGBoost nhanh nhất, XLinear/ST-XLinear chậm nhất.
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Hình 6: Tổng thời gian huấn luyện theo block (phút)', y=1.01)

train_times = raw['training_times']
x = np.arange(len(BLOCKS))
width = 0.12
offsets = np.arange(len(MODELS)) - (len(MODELS) - 1) / 2

for i, model in enumerate(MODELS):
    vals = [train_times[b].get(model, 0) / 60 for b in BLOCKS]  # Convert to minutes
    bars = ax.bar(x + offsets[i] * width, vals, width,
                  label=model, color=MODEL_COLORS[model], alpha=0.85)
    # Annotate
    for bar, v in zip(bars, vals):
        if v > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=7, color=MODEL_COLORS[model])

ax.set_xticks(x)
ax.set_xticklabels([b.replace('block', 'Block ') for b in BLOCKS])
ax.set_ylabel('Thời gian (phút)')
ax.legend(ncol=3, loc='upper center')
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

fig.tight_layout()
savefig(fig, '06_training_time')
plt.close()

print("""
📊 Hình 6 — Thời gian huấn luyện:
   • XGBoost: ~1-2 phút tổng — nhanh nhất nhờ thuật toán histogram-based parallelized.
   • ESTGCN: ~3 phút — gọn nhẹ nhờ kiến trúc GraphConv + LSTM đơn giản.
   • XLinear & ST-XLinear: 40-65 phút — chậm nhất do per-horizon training × per-region.
   • Ensemble: thời gian = thời gian inference XLinear + ESTGCN + grid search (không train mới).
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 9: FIGURE 7 — Improvement of Ensemble vs Best Individual
# Ý nghĩa: Kiểm tra giả thuyết Ensemble luôn cải thiện so với đơn lẻ.
#           Giá trị dương = Ensemble tệ hơn best individual (phản trực giác).
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Hình 7: Δ MAE (Ensemble − Best Individual) — Giá trị âm = Ensemble tốt hơn', y=1.02)

for ax_i, block in enumerate(BLOCKS):
    ax = axes[ax_i]
    improvements = []
    for h in HORIZON_NUMS:
        for region in REGIONS:
            sub = df[(df['block'] == block) & (df['region'] == region) & 
                     (df['metric'] == 'MAE') & (df['horizon_h'] == h)]
            ensemble_mae = sub[sub['model'] == 'Ensemble']['value'].values
            others = sub[sub['model'] != 'Ensemble']['value']
            if len(ensemble_mae) > 0 and len(others) > 0:
                best_other = others.min()
                delta = ensemble_mae[0] - best_other
                improvements.append({'horizon': f'T+{h}', 'region': region, 'delta': delta})
    
    imp_df = pd.DataFrame(improvements)
    if len(imp_df) > 0:
        for region in REGIONS:
            r_data = imp_df[imp_df['region'] == region]
            color = '#58a6ff' if region == 'north' else '#f0883e'
            ax.bar([f"{r['horizon']}\n{region[0].upper()}" for _, r in r_data.iterrows()],
                   r_data['delta'], color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=0, color='#ff7b72', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_ylabel('Δ MAE (µg/m³)' if ax_i == 0 else '')
    ax.set_title(block.replace('block', 'Block '))
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

fig.tight_layout()
savefig(fig, '07_ensemble_improvement')
plt.close()

print("""
📊 Hình 7 — Ensemble Improvement:
   • Ở North, Ensemble hiếm khi thắng XGBoost (delta thường > 0) vì XGBoost quá mạnh ở vùng này.
   • Ở South, Ensemble đôi khi cải thiện nhẹ ở T+1, T+3 nhờ ESTGCN bổ sung spatial info.
   • Kết luận: Ensemble kiểu weighted average đơn giản KHÔNG đảm bảo cải thiện universal.
     Cần kỹ thuật stacking nâng cao hoặc learned weighting mới phát huy hiệu quả.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 10: FIGURE 8 — North vs South Performance Gap
# Ý nghĩa: Phân tích ảnh hưởng địa lý (vùng miền) lên độ chính xác.
#           Gap lớn → model chưa tổng quát hóa tốt giữa các micro-climate.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Hình 8: Chênh lệch MAE giữa South và North (Block 7)', y=1.02)

for ax_i, metric in enumerate(['MAE', 'RMSE']):
    ax = axes[ax_i]
    for model in MODELS:
        north = df[(df['block'] == 'block7') & (df['region'] == 'north') & 
                   (df['model'] == model) & (df['metric'] == metric)]
        south = df[(df['block'] == 'block7') & (df['region'] == 'south') & 
                   (df['model'] == model) & (df['metric'] == metric)]
        if len(north) > 0 and len(south) > 0:
            n_vals = north.sort_values('horizon_h')['value'].values
            s_vals = south.sort_values('horizon_h')['value'].values
            gap = s_vals - n_vals
            ax.plot(HORIZON_NUMS, gap, marker=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model], label=model, linewidth=2, markersize=7)
    
    ax.set_xlabel('Tầm nhìn dự báo (giờ)')
    ax.set_ylabel(f'Δ {metric} (South − North)')
    ax.set_title(f'{metric}: South − North')
    ax.set_xticks(HORIZON_NUMS)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='#8b949e', linestyle=':', alpha=0.5)
    if ax_i == 1:
        ax.legend(loc='upper left', framealpha=0.8)

fig.tight_layout()
savefig(fig, '08_north_south_gap')
plt.close()

print("""
📊 Hình 8 — North vs South Gap:
   • Mọi model đều cho kết quả tệ hơn đáng kể ở South, đặc biệt ở T+12, T+24.
   • XGBoost: gap nhỏ nhất (~0-40 µg/m³) nhờ feature neighbor giúp nắm bắt spatial.
   • XLinear/iTransformer: gap tăng phi tuyến ở horizon dài vì miền Nam có outlier nồng độ PM2.5 cực cao.
   • Gợi ý: cần chiến lược riêng cho South — data augmentation hoặc per-region hyperparameter tuning.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 11: FIGURE 9 — Overall Ranking Summary Table
# Ý nghĩa: Bảng xếp hạng tổng hợp giúp chốt model recommendation cuối cùng.
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  BẢNG TỔNG HỢP — XẾP HẠNG MÔ HÌNH (Block 7, trung bình 2 vùng)")
print("="*80)

summary_rows = []
for model in MODELS:
    row = {'Model': model}
    for h in HORIZON_NUMS:
        for metric in ['MAE', 'RMSE', 'R2']:
            sub = df[(df['block'] == 'block7') & (df['model'] == model) & 
                     (df['metric'] == metric) & (df['horizon_h'] == h)]
            row[f'{metric}_T{h}'] = sub['value'].mean()
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False, float_format='%.2f'))


# ══════════════════════════════════════════════════════════════════════════
# CELL 12: FIGURE 10 — Degradation Rate (MAE increase per horizon hour)
# Ý nghĩa: Tốc độ suy giảm chất lượng dự báo theo giờ.
#           Slope thấp → model giữ phong độ tốt hơn ở dự báo dài hạn.
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Hình 9: Tốc độ suy giảm MAE (slope) — Block 7, trung bình N+S', y=1.01)

slopes = {}
for model in MODELS:
    pivot = weighted_avg('block7', 'MAE')
    if model in pivot.index:
        vals = pivot.loc[model].values
        # Linear regression slope
        slope = np.polyfit(HORIZON_NUMS, vals, 1)[0]
        slopes[model] = slope

sorted_models = sorted(slopes.keys(), key=lambda m: slopes[m])
colors_sorted = [MODEL_COLORS[m] for m in sorted_models]
vals_sorted = [slopes[m] for m in sorted_models]

bars = ax.barh(range(len(sorted_models)), vals_sorted, color=colors_sorted, alpha=0.85,
               edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(sorted_models)))
ax.set_yticklabels(sorted_models)
ax.set_xlabel('Slope (µg/m³ mỗi giờ horizon)')
ax.grid(True, axis='x', linestyle='--', alpha=0.3)

# Annotate
for bar, v in zip(bars, vals_sorted):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=10, color='#c9d1d9')

fig.tight_layout()
savefig(fig, '09_degradation_slope')
plt.close()

print("""
📊 Hình 9 — Degradation Rate:
   • XGBoost: slope ~0.28 µg/m³ per hour — thấp nhất → suy giảm chậm nhất.
   • iTransformer: slope cao nhất → dự báo nhanh chóng mất chất lượng khi horizon tăng.
   • ST-XLinear < XLinear thuần — spatial context giúp giảm tốc độ suy giảm.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 13: FIGURE 11 — Block Comparison: Which split strategy is best?
# Ý nghĩa: So sánh 3 chiến lược chia dữ liệu (5, 7, 30 ngày).
#           Block tốt nhất → dùng cho báo cáo chính thức.
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Hình 10: So sánh 3 chiến lược Block Split — MAE (trên) & R² (dưới)', y=1.02)

for row_i, metric in enumerate(['MAE', 'R2']):
    for col_i, h in enumerate([1, 6, 24]):
        ax = axes[row_i, col_i]
        for model in MODELS:
            vals = []
            for block in BLOCKS:
                sub = df[(df['block'] == block) & (df['model'] == model) & 
                         (df['metric'] == metric) & (df['horizon_h'] == h)]
                vals.append(sub['value'].mean())
            ax.plot(BLOCKS, vals, marker=MODEL_MARKERS[model], color=MODEL_COLORS[model],
                    label=model if (row_i == 0 and col_i == 2) else '', linewidth=2, markersize=7)
        
        ax.set_title(f'{metric} — T+{h}h')
        ax.set_xticklabels([b.replace('block', 'B') for b in BLOCKS])
        ax.grid(True, linestyle='--', alpha=0.3)
        if row_i == 0 and col_i == 2:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.8)

fig.tight_layout()
savefig(fig, '10_block_comparison')
plt.close()

print("""
📊 Hình 10 — Block Split Comparison:
   • Block 7 cho MAE thấp nhất ở hầu hết model × horizon — đặc biệt ở T+1 và T+6.
   • Block 30 tập trung nhiều dữ liệu train liên tục hơn, nhưng val/test ít đa dạng mùa vụ hơn.
   • Block 5 quá chi tiết, dữ liệu train mỗi khối nhỏ, gây underfitting cho DL models.
   • → Khuyến nghị: Sử dụng Block 7 làm chiến lược chính thức cho benchmark.
""")


# ══════════════════════════════════════════════════════════════════════════
# CELL 14: Final Summary
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  KẾT LUẬN PHÂN TÍCH BENCHMARK")
print("="*80)
print("""
1. MÔ HÌNH TỐT NHẤT TỔNG THỂ: XGBoost
   - Dẫn đầu ở mọi horizon, mọi block, mọi region.
   - MAE thấp nhất, R² cao nhất, slope suy giảm thấp nhất.
   - Lý do: Feature engineering thủ công (lag, rolling, KNN neighbor, future weather)
     bù đắp cho kiến trúc đơn giản hơn Deep Learning.

2. MÔ HÌNH DEEP LEARNING TỐT NHẤT: ST-XLinear (đề xuất)
   - Vượt XLinear thuần & iTransformer nhờ spatial context từ Dynamic Graph.
   - Đặc biệt hiệu quả ở tầm nhìn ngắn/trung T+1 → T+6.

3. ENSEMBLE: Hiệu quả hạn chế
   - Weighted average đơn giản chưa đủ khai thác complementarity.
   - Cần nâng cấp: stacking, learned weights, hoặc per-horizon α.

4. ẢNH HƯỞNG ĐỊA LÝ: South luôn khó hơn North
   - PM2.5 phía Nam cao và biến động mạnh → RMSE gap tăng phi tuyến.
   - Gợi ý: Region-specific hyperparameter tuning cần thiết.

5. BLOCK SPLIT: Block 7 khuyến nghị
   - Chu kỳ 7 ngày phản ánh tốt nhất quy luật tuần trong dữ liệu.
""")

print(f"\n✅ Tất cả {len(os.listdir(SAVE_DIR))} biểu đồ đã được lưu tại: report/figures/")
