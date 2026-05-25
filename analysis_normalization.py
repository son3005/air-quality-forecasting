"""
analysis_normalization.py

Phân tích chuyên sâu Pipeline Normalization trong build_dataset.py
Vẽ tất cả biểu đồ cần thiết để chứng minh TẠI SAO lại dùng từng loại normalization
cho từng nhóm features.

Output: Tạo thư mục report/normalization_analysis/ chứa tất cả biểu đồ.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CLEAN_DIR = 'data/clean'
NORM_DIR = 'data/normalized'
OUTPUT_DIR = 'report/normalization_analysis'
STATION_ID = 1  # Trạm mẫu để phân tích

# Nhóm features theo chiến lược normalization (từ build_dataset.py)
NORM_GROUPS = {
    'log1p + StandardScaler': {
        'cols': ['co', 'oxidation_potential', 'pollution_load'],
        'reason': 'Log-normal distribution, moderate outliers',
        'color': '#4e79a7'
    },
    'log1p + RobustScaler': {
        'cols': ['so2', 'humid_sulfate_risk'],
        'reason': 'Log-normal + heavy outliers',
        'color': '#f28e2b'
    },
    'log1p + RobustScaler\n(PM2.5 target, no clip)': {
        'cols': ['pm25'],
        'reason': 'Target variable — preserve extreme values',
        'color': '#e15759'
    },
    'clip(q99) + log1p + MinMaxScaler': {
        'cols': ['precip', 'dust_source_potential'],
        'reason': 'Zero-inflated + extreme spikes',
        'color': '#76b7b2'
    },
    'clip(q99.5) + log1p + RobustScaler': {
        'cols': ['pm10', 'no2'],
        'reason': 'Heavy-tailed + moderate outliers',
        'color': '#59a14f'
    },
    'RobustScaler only': {
        'cols': ['o3', 'wind_spd', 'wind_gusts'],
        'reason': 'Already near-symmetric, some outliers',
        'color': '#edc948'
    },
    'StandardScaler only': {
        'cols': ['temp', 'dewpt', 'thermal_stability', 'soil_temp_0_7', 'no2_so2_log_diff'],
        'reason': 'Near-Gaussian distribution',
        'color': '#b07aa1'
    },
    'MinMaxScaler only': {
        'cols': ['rh', 'clouds', 'soil_moist_0_7'],
        'reason': 'Bounded range [0, 100] or [0, 1]',
        'color': '#ff9da7'
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_data(station_id):
    """Load clean và normalized data cho station."""
    clean_path = f'{CLEAN_DIR}/clean_station_{station_id}.csv'
    norm_path = f'{NORM_DIR}/norm_station_{station_id}.csv'
    
    df_clean = pd.read_csv(clean_path, parse_dates=['timestamp'])
    df_clean = df_clean.set_index('timestamp').sort_index()
    
    df_norm = pd.read_csv(norm_path, parse_dates=['timestamp'])
    df_norm = df_norm.set_index('timestamp').sort_index()
    
    return df_clean, df_norm


def compute_dist_stats(series):
    """Tính các thống kê phân phối cho một cột."""
    s = series.dropna()
    if len(s) < 10:
        return {}
    return {
        'mean': s.mean(),
        'median': s.median(),
        'std': s.std(),
        'skewness': s.skew(),
        'kurtosis': s.kurtosis(),
        'min': s.min(),
        'max': s.max(),
        'q1': s.quantile(0.25),
        'q3': s.quantile(0.75),
        'q99': s.quantile(0.99),
        'q995': s.quantile(0.995),
        'iqr': s.quantile(0.75) - s.quantile(0.25),
        'zero_pct': (s == 0).mean() * 100,
        'n': len(s),
    }


# ============================================================================
# FIGURE 1: Tổng quan — Tại sao cần normalization khác nhau
# ============================================================================

def fig1_overview_distributions(df_clean):
    """Vẽ phân phối raw của TẤT CẢ features để thấy sự đa dạng."""
    
    all_cols = []
    for group in NORM_GROUPS.values():
        all_cols.extend([c for c in group['cols'] if c in df_clean.columns])
    
    n_cols = len(all_cols)
    n_rows = (n_cols + 3) // 4
    
    fig, axes = plt.subplots(n_rows, 4, figsize=(24, n_rows * 4.5))
    fig.suptitle('Hình 1: Phân phối Raw của tất cả Features (TRƯỚC normalization)\n'
                 'Cho thấy tại sao KHÔNG THỂ dùng cùng 1 loại scaler cho mọi feature',
                 fontsize=16, fontweight='bold', y=1.02)
    
    axes = axes.flatten()
    
    for i, col in enumerate(all_cols):
        ax = axes[i]
        data = df_clean[col].dropna()
        
        # Tìm nhóm normalization
        group_name = ''
        color = '#999'
        for gname, ginfo in NORM_GROUPS.items():
            if col in ginfo['cols']:
                group_name = gname
                color = ginfo['color']
                break
        
        # Histogram
        ax.hist(data, bins=80, color=color, alpha=0.7, edgecolor='white', linewidth=0.3, density=True)
        
        # Stats text
        sk = data.skew()
        ku = data.kurtosis()
        zero_pct = (data == 0).mean() * 100
        
        stats_text = f'Skew: {sk:.2f}\nKurt: {ku:.2f}'
        if zero_pct > 1:
            stats_text += f'\nZeros: {zero_pct:.1f}%'
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        
        # Annotation cho nhóm
        ax.text(0.05, 0.95, group_name.split('\n')[0][:25], transform=ax.transAxes,
                fontsize=7, color=color, fontweight='bold', verticalalignment='top')
    
    # Ẩn axes thừa
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig1_overview_distributions.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 1: Overview distributions")


# ============================================================================
# FIGURE 2: So sánh Skewness & Kurtosis — Lý do chọn log transform
# ============================================================================

def fig2_skewness_kurtosis(df_clean):
    """Bar chart so sánh skewness/kurtosis giữa các groups."""
    
    all_cols = []
    col_groups = {}
    col_colors = {}
    for gname, ginfo in NORM_GROUPS.items():
        for c in ginfo['cols']:
            if c in df_clean.columns:
                all_cols.append(c)
                col_groups[c] = gname.split('\n')[0][:30]
                col_colors[c] = ginfo['color']
    
    skews = [df_clean[c].dropna().skew() for c in all_cols]
    kurts = [df_clean[c].dropna().kurtosis() for c in all_cols]
    colors = [col_colors[c] for c in all_cols]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle('Hình 2: Skewness & Kurtosis — Quyết định Log Transform\n'
                 'Skewness > 2 hoặc Kurtosis > 7 → CẦN log1p transform',
                 fontsize=15, fontweight='bold')
    
    # Skewness
    bars1 = ax1.bar(range(len(all_cols)), skews, color=colors, alpha=0.8, edgecolor='white')
    ax1.axhline(y=2, color='red', linestyle='--', linewidth=2, label='Ngưỡng log1p (skew > 2)')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_xticks(range(len(all_cols)))
    ax1.set_xticklabels(all_cols, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Skewness', fontsize=12)
    ax1.set_title('Skewness (đo độ lệch phải)', fontsize=12)
    ax1.legend(fontsize=11)
    
    # Highlight
    for idx, (s, c) in enumerate(zip(skews, all_cols)):
        if s > 2:
            ax1.annotate(f'{s:.1f}', (idx, s), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8, fontweight='bold', color='red')
    
    # Kurtosis
    bars2 = ax2.bar(range(len(all_cols)), kurts, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=7, color='red', linestyle='--', linewidth=2, label='Ngưỡng log1p (kurt > 7)')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(len(all_cols)))
    ax2.set_xticklabels(all_cols, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Kurtosis', fontsize=12)
    ax2.set_title('Kurtosis (đo mức nhọn đỉnh, đuôi nặng)', fontsize=12)
    ax2.legend(fontsize=11)
    
    for idx, (k, c) in enumerate(zip(kurts, all_cols)):
        if k > 7:
            ax2.annotate(f'{k:.1f}', (idx, k), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8, fontweight='bold', color='red')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig2_skewness_kurtosis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 2: Skewness & Kurtosis analysis")


# ============================================================================
# FIGURE 3: Before vs After log1p — Chứng minh hiệu quả
# ============================================================================

def fig3_log_transform_effect(df_clean):
    """So sánh phân phối trước/sau log1p cho các cột cần log."""
    
    log_cols = ['co', 'so2', 'pm25', 'pm10', 'no2', 'precip',
                'oxidation_potential', 'pollution_load', 'humid_sulfate_risk', 'dust_source_potential']
    log_cols = [c for c in log_cols if c in df_clean.columns]
    
    n = len(log_cols)
    fig, axes = plt.subplots(n, 2, figsize=(16, n * 3.5))
    fig.suptitle('Hình 3: Hiệu quả Log1p Transform — Before vs After\n'
                 'Log1p giảm skewness, thu gọn range, đưa phân phối gần Gaussian hơn',
                 fontsize=15, fontweight='bold', y=1.01)
    
    for i, col in enumerate(log_cols):
        data = df_clean[col].dropna().clip(lower=0)
        log_data = np.log1p(data)
        
        # Before
        ax_before = axes[i, 0]
        ax_before.hist(data, bins=80, color='#e15759', alpha=0.7, edgecolor='white', linewidth=0.3, density=True)
        sk_before = data.skew()
        ax_before.set_title(f'{col} — RAW (skew={sk_before:.2f})', fontsize=10, fontweight='bold')
        ax_before.axvline(data.median(), color='black', linestyle='--', linewidth=1, label=f'Median={data.median():.1f}')
        ax_before.legend(fontsize=8)
        
        # After
        ax_after = axes[i, 1]
        ax_after.hist(log_data, bins=80, color='#4e79a7', alpha=0.7, edgecolor='white', linewidth=0.3, density=True)
        sk_after = log_data.skew()
        ax_after.set_title(f'{col} — LOG1P (skew={sk_after:.2f})', fontsize=10, fontweight='bold')
        ax_after.axvline(log_data.median(), color='black', linestyle='--', linewidth=1, label=f'Median={log_data.median():.2f}')
        ax_after.legend(fontsize=8)
        
        # Improvement annotation
        improvement = abs(sk_before) - abs(sk_after)
        if improvement > 0:
            ax_after.text(0.95, 0.85, f'Skew giảm {improvement:.1f}', transform=ax_after.transAxes,
                         fontsize=9, color='green', fontweight='bold', ha='right',
                         bbox=dict(facecolor='lightgreen', alpha=0.5, boxstyle='round'))
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig3_log_transform_effect.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 3: Log transform effect")


# ============================================================================
# FIGURE 4: Tại sao chọn RobustScaler vs StandardScaler vs MinMaxScaler
# ============================================================================

def fig4_scaler_comparison(df_clean):
    """So sánh 3 scalers trên cùng feature để thấy ưu/nhược."""
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    
    # Chọn representative features cho mỗi case
    test_cases = {
        'pm25\n(có outliers cực đoan)': 'pm25',
        'temp\n(gần Gaussian)': 'temp',
        'rh\n(bounded [0,100])': 'rh',
    }
    
    fig, axes = plt.subplots(len(test_cases), 4, figsize=(24, len(test_cases) * 4.5))
    fig.suptitle('Hình 4: So sánh 3 Scalers trên các loại phân phối khác nhau\n'
                 'Thể hiện TẠI SAO mỗi nhóm feature cần scaler riêng',
                 fontsize=15, fontweight='bold', y=1.02)
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler(),
    }
    scaler_colors = {
        'StandardScaler': '#b07aa1',
        'RobustScaler': '#f28e2b',
        'MinMaxScaler': '#ff9da7',
    }
    
    for i, (label, col) in enumerate(test_cases.items()):
        data = df_clean[col].dropna().values.reshape(-1, 1)
        
        # Original
        axes[i, 0].hist(data, bins=60, color='gray', alpha=0.7, density=True, edgecolor='white')
        axes[i, 0].set_title(f'{label}\nOriginal', fontsize=10, fontweight='bold')
        
        best_scaler = ''
        best_reason = ''
        
        for j, (sname, scaler) in enumerate(scalers.items()):
            sc = type(scaler)()  # Create fresh instance
            scaled = sc.fit_transform(data)
            
            ax = axes[i, j + 1]
            ax.hist(scaled, bins=60, color=scaler_colors[sname], alpha=0.7, density=True, edgecolor='white')
            
            sk = pd.Series(scaled.flatten()).skew()
            rng = scaled.max() - scaled.min()
            
            ax.set_title(f'{sname}\nskew={sk:.2f}, range={rng:.2f}', fontsize=10)
            
            # Mark the "best fit" based on feature type
            if col == 'pm25' and sname == 'RobustScaler':
                ax.set_title(f'✅ {sname}\nskew={sk:.2f}, range={rng:.2f}', fontsize=10, color='green', fontweight='bold')
            elif col == 'temp' and sname == 'StandardScaler':
                ax.set_title(f'✅ {sname}\nskew={sk:.2f}, range={rng:.2f}', fontsize=10, color='green', fontweight='bold')
            elif col == 'rh' and sname == 'MinMaxScaler':
                ax.set_title(f'✅ {sname}\nskew={sk:.2f}, range={rng:.2f}', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig4_scaler_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 4: Scaler comparison")


# ============================================================================
# FIGURE 5: Outlier Analysis — Box plots trước/sau normalization
# ============================================================================

def fig5_outlier_boxplots(df_clean, df_norm):
    """Box plots for/after normalization — thấy outlier bị xử lý thế nào."""
    
    # Chọn features đại diện
    features = ['pm25', 'co', 'so2', 'no2', 'pm10', 'o3', 'temp', 'rh', 'precip']
    features = [f for f in features if f in df_clean.columns and f in df_norm.columns]
    
    fig, axes = plt.subplots(2, len(features), figsize=(len(features) * 3, 10))
    fig.suptitle('Hình 5: Box Plots — Trước vs Sau Normalization\n'
                 'Thấy rõ outliers được xử lý (thu gọn range) qua normalization',
                 fontsize=15, fontweight='bold', y=1.02)
    
    for i, col in enumerate(features):
        # Before
        ax_b = axes[0, i]
        data_b = df_clean[col].dropna()
        bp1 = ax_b.boxplot(data_b, patch_artist=True, 
                           boxprops=dict(facecolor='#e15759', alpha=0.6),
                           medianprops=dict(color='black', linewidth=2))
        ax_b.set_title(f'{col}\nBEFORE', fontsize=9, fontweight='bold')
        n_outliers_before = ((data_b < data_b.quantile(0.25) - 1.5 * (data_b.quantile(0.75) - data_b.quantile(0.25))) |
                            (data_b > data_b.quantile(0.75) + 1.5 * (data_b.quantile(0.75) - data_b.quantile(0.25)))).sum()
        ax_b.text(0.5, -0.15, f'Outliers: {n_outliers_before}', transform=ax_b.transAxes,
                 ha='center', fontsize=8, color='red')
        
        # After
        ax_a = axes[1, i]
        data_a = df_norm[col].dropna()
        bp2 = ax_a.boxplot(data_a, patch_artist=True,
                           boxprops=dict(facecolor='#4e79a7', alpha=0.6),
                           medianprops=dict(color='black', linewidth=2))
        ax_a.set_title(f'{col}\nAFTER', fontsize=9, fontweight='bold')
        n_outliers_after = ((data_a < data_a.quantile(0.25) - 1.5 * (data_a.quantile(0.75) - data_a.quantile(0.25))) |
                           (data_a > data_a.quantile(0.75) + 1.5 * (data_a.quantile(0.75) - data_a.quantile(0.25)))).sum()
        ax_a.text(0.5, -0.15, f'Outliers: {n_outliers_after}', transform=ax_a.transAxes,
                 ha='center', fontsize=8, color='blue')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig5_outlier_boxplots.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 5: Outlier boxplots")


# ============================================================================
# FIGURE 6: Zero-Inflated Features — Tại sao precip cần clip+log+MinMax
# ============================================================================

def fig6_zero_inflated(df_clean):
    """Phân tích precip và dust_source_potential — zero-inflated distributions."""
    
    cols = ['precip', 'dust_source_potential']
    cols = [c for c in cols if c in df_clean.columns]
    
    fig, axes = plt.subplots(len(cols), 3, figsize=(20, len(cols) * 5))
    if len(cols) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Hình 6: Zero-Inflated Features — Tại sao cần clip(q99) + log1p + MinMaxScaler\n'
                 'Features này có >50% giá trị = 0, cùng với các spike cực đoan',
                 fontsize=15, fontweight='bold', y=1.02)
    
    for i, col in enumerate(cols):
        data = df_clean[col].dropna()
        zero_pct = (data == 0).mean() * 100
        q99 = data.quantile(0.99)
        
        # Original
        ax1 = axes[i, 0]
        ax1.hist(data, bins=100, color='#76b7b2', alpha=0.7, edgecolor='white', density=True)
        ax1.set_title(f'{col} — Original\nZeros: {zero_pct:.1f}%, Range: [0, {data.max():.2f}]', 
                     fontsize=10, fontweight='bold')
        ax1.axvline(q99, color='red', linestyle='--', label=f'q99={q99:.3f}')
        ax1.legend(fontsize=9)
        
        # After clip + log1p
        clipped = np.log1p(data.clip(lower=0, upper=q99))
        ax2 = axes[i, 1]
        ax2.hist(clipped, bins=100, color='#59a14f', alpha=0.7, edgecolor='white', density=True)
        ax2.set_title(f'{col} — Clip(q99) + Log1p\nskew: {data.skew():.1f} → {clipped.skew():.1f}',
                     fontsize=10, fontweight='bold')
        
        # After full pipeline
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        final = sc.fit_transform(clipped.values.reshape(-1, 1)).flatten()
        ax3 = axes[i, 2]
        ax3.hist(final, bins=100, color='#4e79a7', alpha=0.7, edgecolor='white', density=True)
        ax3.set_title(f'{col} — Full Pipeline\nRange: [{final.min():.2f}, {final.max():.2f}]',
                     fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig6_zero_inflated.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 6: Zero-inflated analysis")


# ============================================================================
# FIGURE 7: PM2.5 — Target Variable Special Treatment
# ============================================================================

def fig7_pm25_target_analysis(df_clean, df_norm):
    """Phân tích đặc biệt cho PM2.5 — target variable."""
    
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Hình 7: PM2.5 (Target Variable) — Tại sao dùng log1p + RobustScaler KHÔNG CLIP\n'
                 'Giữ nguyên extreme values vì đó là thông tin quan trọng nhất cho dự báo ô nhiễm',
                 fontsize=14, fontweight='bold', y=1.02)
    
    pm25_raw = df_clean['pm25'].dropna()
    pm25_norm = df_norm['pm25'].dropna()
    
    # 1. Raw distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pm25_raw, bins=100, color='#e15759', alpha=0.7, edgecolor='white', density=True)
    ax1.set_title(f'Raw PM2.5\nskew={pm25_raw.skew():.2f}, kurt={pm25_raw.kurtosis():.2f}', fontweight='bold')
    ax1.axvline(75, color='orange', linestyle='--', linewidth=2, label='WHO Unhealthy (75 µg/m³)')
    ax1.axvline(150, color='red', linestyle='--', linewidth=2, label='Hazardous (150 µg/m³)')
    ax1.legend(fontsize=8)
    
    # 2. After log1p
    pm25_log = np.log1p(pm25_raw.clip(lower=0))
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(pm25_log, bins=100, color='#f28e2b', alpha=0.7, edgecolor='white', density=True)
    ax2.set_title(f'After log1p\nskew={pm25_log.skew():.2f}, kurt={pm25_log.kurtosis():.2f}', fontweight='bold')
    
    # 3. After log1p + RobustScaler (final)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(pm25_norm, bins=100, color='#4e79a7', alpha=0.7, edgecolor='white', density=True)
    ax3.set_title(f'Final (log1p + RobustScaler)\nskew={pm25_norm.skew():.2f}, range=[{pm25_norm.min():.1f}, {pm25_norm.max():.1f}]', fontweight='bold')
    
    # 4. Time series trước/sau
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(pm25_raw.index[:2000], pm25_raw.values[:2000], color='#e15759', alpha=0.6, linewidth=0.5, label='Raw')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(pm25_norm.index[:2000], pm25_norm.values[:2000], color='#4e79a7', alpha=0.6, linewidth=0.5, label='Normalized')
    ax4.set_title('Time Series: Raw vs Normalized (2000 giờ đầu)', fontweight='bold')
    ax4.set_ylabel('Raw PM2.5 (µg/m³)', color='#e15759')
    ax4_twin.set_ylabel('Normalized PM2.5', color='#4e79a7')
    ax4.legend(loc='upper left', fontsize=9)
    ax4_twin.legend(loc='upper right', fontsize=9)
    
    # 5. Q-Q plot
    ax5 = fig.add_subplot(gs[1, 2])
    stats.probplot(pm25_norm.values, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot (Normalized PM2.5 vs Normal)', fontweight='bold')
    ax5.get_lines()[0].set_color('#4e79a7')
    ax5.get_lines()[1].set_color('red')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig7_pm25_target_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 7: PM2.5 target analysis")


# ============================================================================
# FIGURE 8: Bounded Features — Tại sao dùng MinMaxScaler
# ============================================================================

def fig8_bounded_features(df_clean, df_norm):
    """Phân tích rh, clouds, soil_moist — bounded trong range cố định."""
    
    cols = ['rh', 'clouds', 'soil_moist_0_7']
    cols = [c for c in cols if c in df_clean.columns]
    
    fig, axes = plt.subplots(len(cols), 3, figsize=(20, len(cols) * 4.5))
    fig.suptitle('Hình 8: Bounded Features — Tại sao dùng MinMaxScaler\n'
                 'Các features đã có range tự nhiên → MinMaxScaler giữ nguyên shape, chỉ rescale',
                 fontsize=15, fontweight='bold', y=1.02)
    
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    
    for i, col in enumerate(cols):
        data = df_clean[col].dropna().values.reshape(-1, 1)
        bounds = {'rh': '[0, 100]%', 'clouds': '[0, 100]%', 'soil_moist_0_7': '[0, ~0.5] m³/m³'}
        
        # Original
        axes[i, 0].hist(data, bins=60, color='#ff9da7', alpha=0.7, edgecolor='white', density=True)
        axes[i, 0].set_title(f'{col} — Original\nNatural range: {bounds.get(col, "?")}', fontsize=10, fontweight='bold')
        
        # StandardScaler (NOT ideal — may exceed [0,1])
        std_scaled = StandardScaler().fit_transform(data)
        axes[i, 1].hist(std_scaled, bins=60, color='#b07aa1', alpha=0.7, edgecolor='white', density=True)
        axes[i, 1].set_title(f'❌ StandardScaler\nRange: [{std_scaled.min():.2f}, {std_scaled.max():.2f}]', fontsize=10)
        
        # MinMaxScaler (ideal)
        mm_scaled = MinMaxScaler().fit_transform(data)
        axes[i, 2].hist(mm_scaled, bins=60, color='#4e79a7', alpha=0.7, edgecolor='white', density=True)
        axes[i, 2].set_title(f'✅ MinMaxScaler\nRange: [{mm_scaled.min():.2f}, {mm_scaled.max():.2f}]', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig8_bounded_features.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 8: Bounded features")


# ============================================================================
# FIGURE 9: Gaussian Features — Tại sao dùng StandardScaler
# ============================================================================

def fig9_gaussian_features(df_clean, df_norm):
    """Chứng minh temp, dewpt, thermal_stability gần Gaussian."""
    
    cols = ['temp', 'dewpt', 'thermal_stability', 'soil_temp_0_7', 'no2_so2_log_diff']
    cols = [c for c in cols if c in df_clean.columns]
    
    fig, axes = plt.subplots(len(cols), 2, figsize=(16, len(cols) * 4))
    fig.suptitle('Hình 9: Gaussian Features — Tại sao dùng StandardScaler\n'
                 'Các features này có phân phối gần chuẩn (|skew| < 1), StandardScaler là optimal',
                 fontsize=15, fontweight='bold', y=1.02)
    
    for i, col in enumerate(cols):
        data = df_clean[col].dropna()
        
        # Histogram + fitted normal
        ax1 = axes[i, 0]
        ax1.hist(data, bins=80, color='#b07aa1', alpha=0.7, edgecolor='white', density=True)
        
        # Fit normal distribution
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 200)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal fit (μ={mu:.1f}, σ={sigma:.1f})')
        ax1.set_title(f'{col} — Histogram + Normal Fit\nskew={data.skew():.2f}', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        
        # Q-Q plot
        ax2 = axes[i, 1]
        stats.probplot(data.values, dist="norm", plot=ax2)
        ax2.set_title(f'{col} — Q-Q Plot', fontsize=10, fontweight='bold')
        ax2.get_lines()[0].set_color('#b07aa1')
        ax2.get_lines()[1].set_color('red')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig9_gaussian_features.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 9: Gaussian features")


# ============================================================================
# FIGURE 10: Clip Effect — pm10, no2 clip ở q99.5
# ============================================================================

def fig10_clip_effect(df_clean):
    """Chứng minh hiệu quả clip ở q99/q99.5 trước log transform."""
    
    cols = {'pm10': 0.995, 'no2': 0.995, 'precip': 0.99, 'dust_source_potential': 0.99}
    cols = {c: q for c, q in cols.items() if c in df_clean.columns}
    
    fig, axes = plt.subplots(len(cols), 3, figsize=(20, len(cols) * 4.5))
    fig.suptitle('Hình 10: Hiệu quả Clipping — Cắt extreme values ở quantile cao\n'
                 'Clip giảm range data mà chỉ ảnh hưởng 0.5-1% mẫu cực đoan',
                 fontsize=15, fontweight='bold', y=1.02)
    
    for i, (col, q_thresh) in enumerate(cols.items()):
        data = df_clean[col].dropna().clip(lower=0)
        q_val = data.quantile(q_thresh)
        n_affected = (data > q_val).sum()
        pct_affected = (data > q_val).mean() * 100
        
        # Original
        axes[i, 0].hist(data, bins=100, color='#e15759', alpha=0.7, edgecolor='white', density=True)
        axes[i, 0].axvline(q_val, color='blue', linestyle='--', linewidth=2, label=f'q{q_thresh*100:.1f}={q_val:.2f}')
        axes[i, 0].set_title(f'{col} — Original\nRange: [{data.min():.1f}, {data.max():.1f}]', fontweight='bold', fontsize=10)
        axes[i, 0].legend(fontsize=8)
        
        # After clip
        clipped = data.clip(upper=q_val)
        axes[i, 1].hist(clipped, bins=100, color='#f28e2b', alpha=0.7, edgecolor='white', density=True)
        axes[i, 1].set_title(f'{col} — After clip(q{q_thresh*100:.1f})\nAffected: {n_affected} samples ({pct_affected:.2f}%)',
                            fontweight='bold', fontsize=10)
        
        # After clip + log1p
        log_clipped = np.log1p(clipped)
        axes[i, 2].hist(log_clipped, bins=100, color='#4e79a7', alpha=0.7, edgecolor='white', density=True)
        sk = log_clipped.skew()
        axes[i, 2].set_title(f'{col} — Clip + Log1p\nskew={sk:.2f}', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig10_clip_effect.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 10: Clip effect")


# ============================================================================
# FIGURE 11: Normalization Summary Table — Tổng hợp
# ============================================================================

def fig11_summary_table(df_clean, df_norm):
    """Bảng tổng hợp trước/sau normalization cho TẤT CẢ features."""
    
    rows = []
    for gname, ginfo in NORM_GROUPS.items():
        for col in ginfo['cols']:
            if col in df_clean.columns and col in df_norm.columns:
                raw_stats = compute_dist_stats(df_clean[col])
                norm_stats = compute_dist_stats(df_norm[col])
                
                rows.append({
                    'Feature': col,
                    'Strategy': gname.replace('\n', ' '),
                    'Raw Skew': raw_stats.get('skewness', None),
                    'Norm Skew': norm_stats.get('skewness', None),
                    'Raw Kurt': raw_stats.get('kurtosis', None),
                    'Norm Kurt': norm_stats.get('kurtosis', None),
                    'Raw Range': f"[{raw_stats.get('min', 0):.1f}, {raw_stats.get('max', 0):.1f}]",
                    'Norm Range': f"[{norm_stats.get('min', 0):.2f}, {norm_stats.get('max', 0):.2f}]",
                    'Zeros%': raw_stats.get('zero_pct', 0),
                })
    
    df_summary = pd.DataFrame(rows)
    
    # Save as CSV
    df_summary.to_csv(f'{OUTPUT_DIR}/normalization_summary.csv', index=False)
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(24, max(10, len(rows) * 0.55 + 3)))
    ax.axis('off')
    ax.set_title('Hình 11: Bảng Tổng hợp Normalization — Tất cả Features\n'
                 'So sánh Skewness, Kurtosis và Range trước/sau normalization',
                 fontsize=14, fontweight='bold', pad=20)
    
    col_labels = ['Feature', 'Strategy', 'Raw Skew', 'Norm Skew', 'Raw Kurt', 'Norm Kurt', 'Raw Range', 'Norm Range', 'Zeros%']
    cell_data = []
    for _, row in df_summary.iterrows():
        cell_data.append([
            row['Feature'],
            row['Strategy'][:35],
            f"{row['Raw Skew']:.2f}" if pd.notna(row['Raw Skew']) else 'N/A',
            f"{row['Norm Skew']:.2f}" if pd.notna(row['Norm Skew']) else 'N/A',
            f"{row['Raw Kurt']:.1f}" if pd.notna(row['Raw Kurt']) else 'N/A',
            f"{row['Norm Kurt']:.1f}" if pd.notna(row['Norm Kurt']) else 'N/A',
            row['Raw Range'],
            row['Norm Range'],
            f"{row['Zeros%']:.1f}",
        ])
    
    table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center',
                     cellLoc='center', colColours=['#4e79a7'] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    
    # Color header text
    for j in range(len(col_labels)):
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # Color rows by group
    group_colors = {}
    for gname, ginfo in NORM_GROUPS.items():
        for c in ginfo['cols']:
            group_colors[c] = ginfo['color']
    
    for i in range(len(cell_data)):
        feat = cell_data[i][0]
        color = group_colors.get(feat, '#ffffff')
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color + '20')  # Very light tint
    
    fig.savefig(f'{OUTPUT_DIR}/fig11_summary_table.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 11: Summary table")
    
    return df_summary


# ============================================================================
# FIGURE 12: Wind Direction — Tại sao cần sin/cos decomposition
# ============================================================================

def fig12_wind_direction(df_clean):
    """Chứng minh vấn đề 0°/360° discontinuity và hiệu quả sin/cos."""
    
    if 'wind_dir' not in df_clean.columns:
        print("  ⚠ Skipping fig12 — wind_dir not in clean data")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Hình 12: Wind Direction — Tại sao chuyển sang sin/cos\n'
                 '0° và 360° giống nhau (gió Bắc) nhưng Scaler coi là giá trị MAX-MIN → sai',
                 fontsize=14, fontweight='bold', y=1.05)
    
    wind = df_clean['wind_dir'].dropna()
    
    # Original distribution
    axes[0].hist(wind, bins=72, color='#edc948', alpha=0.7, edgecolor='white')
    axes[0].set_title(f'Wind Direction (°)\nRange: [0, 360] — Circular!', fontweight='bold')
    axes[0].set_xlabel('Degrees')
    axes[0].axvline(0, color='red', linewidth=2, linestyle='--', label='0° = North')
    axes[0].axvline(360, color='red', linewidth=2, linestyle='--', label='360° = North')
    axes[0].legend(fontsize=9)
    
    # Sin component
    rad = np.deg2rad(wind)
    wind_sin = np.sin(rad)
    axes[1].hist(wind_sin, bins=80, color='#4e79a7', alpha=0.7, edgecolor='white')
    axes[1].set_title(f'wind_sin = sin(wind_dir)\nRange: [-1, 1] — Continuous!', fontweight='bold')
    
    # Cos component
    wind_cos = np.cos(rad)
    axes[2].hist(wind_cos, bins=80, color='#59a14f', alpha=0.7, edgecolor='white')
    axes[2].set_title(f'wind_cos = cos(wind_dir)\nRange: [-1, 1] — Continuous!', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig12_wind_direction.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 12: Wind direction analysis")


# ============================================================================
# FIGURE 13: Correlation Heatmap — Before vs After
# ============================================================================

def fig13_correlation_comparison(df_clean, df_norm):
    """So sánh ma trận tương quan trước/sau normalization."""
    
    # Key features only
    key_cols = ['pm25', 'pm10', 'co', 'no2', 'so2', 'o3', 'temp', 'rh', 'wind_spd', 'precip']
    key_cols_clean = [c for c in key_cols if c in df_clean.columns]
    key_cols_norm = [c for c in key_cols if c in df_norm.columns]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Hình 13: Ma trận Tương quan — Trước vs Sau Normalization\n'
                 'Normalization BẢO TOÀN cấu trúc tương quan (correlation preservation)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Before
    corr_before = df_clean[key_cols_clean].corr()
    im1 = ax1.imshow(corr_before, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(key_cols_clean)))
    ax1.set_yticks(range(len(key_cols_clean)))
    ax1.set_xticklabels(key_cols_clean, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(key_cols_clean, fontsize=9)
    ax1.set_title('BEFORE Normalization', fontweight='bold', fontsize=12)
    
    for y in range(len(key_cols_clean)):
        for x in range(len(key_cols_clean)):
            ax1.text(x, y, f'{corr_before.iloc[y, x]:.2f}', ha='center', va='center', fontsize=7)
    
    # After
    corr_after = df_norm[key_cols_norm].corr()
    im2 = ax2.imshow(corr_after, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(key_cols_norm)))
    ax2.set_yticks(range(len(key_cols_norm)))
    ax2.set_xticklabels(key_cols_norm, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(key_cols_norm, fontsize=9)
    ax2.set_title('AFTER Normalization', fontweight='bold', fontsize=12)
    
    for y in range(len(key_cols_norm)):
        for x in range(len(key_cols_norm)):
            ax2.text(x, y, f'{corr_after.iloc[y, x]:.2f}', ha='center', va='center', fontsize=7)
    
    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label='Pearson Correlation')
    
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig13_correlation_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 13: Correlation comparison")


# ============================================================================
# FIGURE 14: Normalization Decision Flowchart (text-based)
# ============================================================================

def fig14_decision_flowchart():
    """Tạo flowchart logic quyết định normalization strategy."""
    
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.axis('off')
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    
    fig.suptitle('Hình 14: Flowchart Quyết định — Chọn Normalization Strategy\n'
                 'Logic chọn scaler dựa trên đặc tính phân phối của từng feature',
                 fontsize=15, fontweight='bold', y=0.98)
    
    def draw_box(x, y, w, h, text, color, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                            facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=fontsize, fontweight='bold', wrap=True)
    
    def draw_arrow(x1, y1, x2, y2, label='', color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, label, fontsize=8, ha='center', color=color, fontweight='bold')
    
    # Start
    draw_box(8.5, 14.5, 5, 1, 'Feature cần normalize', '#ddd', 11)
    
    # Decision 1: Is it bounded?
    draw_box(8.5, 12.5, 5, 1, 'Có range tự nhiên\ncố định? (0-100%)', '#fff3cd')
    draw_arrow(11, 14.5, 11, 13.5)
    
    # Yes → MinMaxScaler
    draw_box(16, 12.5, 5, 1, '✅ MinMaxScaler\nrh, clouds, soil_moist', '#ff9da7')
    draw_arrow(13.5, 13, 16, 13, 'Có')
    
    # No → Decision 2
    draw_box(8.5, 10.5, 5, 1, 'Phân phối gần Gaussian?\n(|skew| < 1)', '#fff3cd')
    draw_arrow(11, 12.5, 11, 11.5, 'Không')
    
    # Yes → StandardScaler
    draw_box(16, 10.5, 5, 1, '✅ StandardScaler\ntemp, dewpt, thermal_stab\nsoil_temp, no2_so2_log', '#b07aa1')
    draw_arrow(13.5, 11, 16, 11, 'Có')
    
    # No → Decision 3
    draw_box(8.5, 8.5, 5, 1, 'Skewness > 2?\n(cần log transform)', '#fff3cd')
    draw_arrow(11, 10.5, 11, 9.5, 'Không')
    
    # No skew → RobustScaler only
    draw_box(16, 8.5, 5, 1, '✅ RobustScaler only\no3, wind_spd, wind_gusts', '#edc948')
    draw_arrow(13.5, 9, 16, 9, 'Không\n(nhưng có outliers)')
    
    # Yes → Decision 4: zeros > 50%?
    draw_box(3, 6.5, 5, 1, 'Zero-inflated?\n(>50% zeros)', '#fff3cd')
    draw_arrow(11, 8.5, 5.5, 7.5, 'Có')
    
    # Yes → clip + log + MinMax
    draw_box(0.5, 4.5, 5, 1, '✅ clip(q99) + log1p\n+ MinMaxScaler\nprecip, dust_source', '#76b7b2')
    draw_arrow(5.5, 6.5, 3, 5.5, 'Có')
    
    # No → Decision 5
    draw_box(9, 6.5, 5, 1, 'Target variable?\n(PM2.5)', '#fff3cd')
    draw_arrow(8, 7, 9, 7, ' Không')
    
    # Yes → log + Robust NO clip
    draw_box(9, 4.5, 5, 1, '✅ log1p + RobustScaler\n(KHÔNG clip)\npm25, pm25_lags', '#e15759')
    draw_arrow(11.5, 6.5, 11.5, 5.5, 'Có')
    
    # No → Decision 6
    draw_box(15, 6.5, 5.5, 1, 'Heavy outliers?\n(cần clip)', '#fff3cd')
    draw_arrow(14, 7, 15, 7, 'Không')
    
    draw_box(15, 4.5, 5.5, 1, '✅ clip(q99.5) + log1p\n+ RobustScaler\npm10, no2', '#59a14f')
    draw_arrow(17.75, 6.5, 17.75, 5.5, 'Có')
    
    # Lower branches for log + standard and log + robust
    draw_box(3, 2.5, 5, 1, '✅ log1p + StandardScaler\nco, oxidation_pot\npollution_load', '#4e79a7')
    draw_box(9, 2.5, 5, 1, '✅ log1p + RobustScaler\nso2, humid_sulfate', '#f28e2b')
    
    draw_arrow(5.5, 4.5, 5.5, 3.5, 'Moderate\noutliers')
    draw_arrow(11.5, 4.5, 11.5, 3.5, 'Heavy\noutliers')
    
    fig.savefig(f'{OUTPUT_DIR}/fig14_decision_flowchart.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ Fig 14: Decision flowchart")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  NORMALIZATION ANALYSIS — Station", STATION_ID)
    print("=" * 60)
    
    print("\n📂 Loading data...")
    df_clean, df_norm = load_data(STATION_ID)
    print(f"  Clean: {df_clean.shape[0]} rows × {df_clean.shape[1]} cols")
    print(f"  Norm:  {df_norm.shape[0]} rows × {df_norm.shape[1]} cols")
    
    print(f"\n📊 Generating figures to {OUTPUT_DIR}/...\n")
    
    fig1_overview_distributions(df_clean)
    fig2_skewness_kurtosis(df_clean)
    fig3_log_transform_effect(df_clean)
    fig4_scaler_comparison(df_clean)
    fig5_outlier_boxplots(df_clean, df_norm)
    fig6_zero_inflated(df_clean)
    fig7_pm25_target_analysis(df_clean, df_norm)
    fig8_bounded_features(df_clean, df_norm)
    fig9_gaussian_features(df_clean, df_norm)
    fig10_clip_effect(df_clean)
    fig11_summary_table(df_clean, df_norm)
    fig12_wind_direction(df_clean)
    fig13_correlation_comparison(df_clean, df_norm)
    fig14_decision_flowchart()
    
    print(f"\n✅ Done! {14} figures saved to {OUTPUT_DIR}/")
    print("=" * 60)
