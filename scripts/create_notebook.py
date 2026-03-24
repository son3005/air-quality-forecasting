import json

cells = []

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

# ============================================================
# CELL 0: TITLE
# ============================================================
cells.append(md([
    "# 📊 Phân Tích & So Sánh Toàn Diện Các Mô Hình PM2.5\n",
    "Notebook này phân tích **có hệ thống** hiệu năng của 6 mô hình trên 3 Data Blocks khác nhau, qua 5 góc nhìn trực quan.\n\n",
    "| Góc nhìn | Mục đích |\n",
    "|---|---|\n",
    "| 1. Heatmap Tổng Hợp | Nhìn 1 phát biết model nào tốt nhất |\n",
    "| 2. Grouped Bar Chart | So sánh tại từng Horizon cụ thể |\n",
    "| 3. Radar Chart | Đánh giá sức mạnh tổng thể |\n",
    "| 4. Box Plot Stability | Đo độ ổn định xuyên Block |\n",
    "| 5. Line Plot Chi Tiết | Phân tích sâu Bắc/Nam/TB từng Block |"
]))

# ============================================================
# CELL 1: IMPORTS & DATA LOADING
# ============================================================
cells.append(code([
    "import json, warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "import seaborn as sns\n",
    "from math import pi\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "plt.rcParams.update({\n",
    "    'figure.dpi': 120,\n",
    "    'font.size': 11,\n",
    "    'axes.titlesize': 14,\n",
    "    'axes.labelsize': 12,\n",
    "    'legend.fontsize': 10,\n",
    "    'figure.facecolor': 'white'\n",
    "})\n",
    "\n",
    "with open('../report/benchmark_metrics.json', 'r') as f:\n",
    "    raw = json.load(f)\n",
    "\n",
    "blocks = list(raw['data'].keys())\n",
    "models = raw['models']\n",
    "horizons = raw['horizons']\n",
    "metrics = raw['metrics']\n",
    "regions = ['north', 'south', 'aggregated']\n",
    "region_vn = {'north': 'Miền Bắc', 'south': 'Miền Nam', 'aggregated': 'Trung Bình'}\n",
    "metric_better = {'RMSE': 'lower', 'MAE': 'lower', 'R2': 'higher'}\n",
    "\n",
    "# Flatten data thành DataFrame để tiện thao tác\n",
    "rows = []\n",
    "for block in blocks:\n",
    "    for region in regions:\n",
    "        for metric in metrics:\n",
    "            for model in models:\n",
    "                vals = raw['data'][block][region][metric][model]\n",
    "                for i, h in enumerate(horizons):\n",
    "                    rows.append({\n",
    "                        'block': block, 'region': region,\n",
    "                        'metric': metric, 'model': model,\n",
    "                        'horizon': h, 'value': vals[i]\n",
    "                    })\n",
    "\n",
    "df = pd.DataFrame(rows)\n",
    "print(f'✅ Loaded {len(df)} records | {len(blocks)} blocks | {len(models)} models')\n",
    "df.head()"
]))

# ============================================================
# PART 1: HEATMAP
# ============================================================
cells.append(md([
    "---\n",
    "## 1️⃣ HEATMAP TỔNG HỢP — Nhìn 1 phát, biết hết\n",
    "Mỗi ô hiển thị giá trị **Aggregated** (trung bình có trọng số). Trục Y = `Model × Block`, Trục X = `Horizon`.\n",
    "- 🟢 Xanh = Tốt hơn (MAE/RMSE thấp hoặc R² cao)\n",
    "- 🔴 Đỏ = Kém hơn"
]))

cells.append(code([
    "fig, axes = plt.subplots(1, 3, figsize=(24, 8))\n",
    "\n",
    "for ax_idx, metric in enumerate(metrics):\n",
    "    sub = df[(df['region'] == 'aggregated') & (df['metric'] == metric)]\n",
    "    pivot = sub.pivot_table(index=['model', 'block'], columns='horizon', values='value')\n",
    "    pivot = pivot[horizons]  # đảm bảo thứ tự cột\n",
    "    \n",
    "    # Sắp xếp theo giá trị tốt nhất (trung bình qua các horizon)\n",
    "    ascending = True if metric_better[metric] == 'lower' else False\n",
    "    pivot['_mean'] = pivot.mean(axis=1)\n",
    "    pivot = pivot.sort_values('_mean', ascending=ascending)\n",
    "    pivot = pivot.drop(columns='_mean')\n",
    "    \n",
    "    # Chọn colormap phù hợp\n",
    "    cmap = 'RdYlGn' if metric == 'R2' else 'RdYlGn_r'\n",
    "    \n",
    "    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5,\n",
    "                ax=axes[ax_idx], cbar_kws={'shrink': 0.8})\n",
    "    axes[ax_idx].set_title(f'Heatmap {metric} (Aggregated)', fontsize=15, fontweight='bold')\n",
    "    axes[ax_idx].set_ylabel('')\n",
    "    axes[ax_idx].set_xlabel('Horizon')\n",
    "\n",
    "plt.suptitle('HEATMAP TỔNG HỢP — So sánh nhanh Model × Block × Horizon', fontsize=17, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md([
    "**📝 Cách đọc Heatmap:**\n",
    "- Nhìn từ trên xuống dưới theo cột: Model + Block nào có ô xanh nhất → Tốt nhất.\n",
    "- So sánh ngang (cùng 1 Model): Nếu các ô thay đổi màu ít giữa Block → Model ổn định.\n",
    "- So sánh dọc (cùng 1 Horizon): Ô xanh nhất ở dưới cùng = Best model cho horizon đó."
]))

# ============================================================
# PART 2: GROUPED BAR CHART
# ============================================================
cells.append(md([
    "---\n",
    "## 2️⃣ GROUPED BAR CHART — Tại mỗi Horizon, ai thắng?\n",
    "Với mỗi Horizon, nhóm các thanh Bar cạnh nhau (1 thanh = 1 Model), chia theo Block."
]))

cells.append(code([
    "for metric in metrics:\n",
    "    sub = df[(df['region'] == 'aggregated') & (df['metric'] == metric)]\n",
    "    \n",
    "    fig, axes = plt.subplots(1, len(blocks), figsize=(7 * len(blocks), 6), sharey=True)\n",
    "    if len(blocks) == 1:\n",
    "        axes = [axes]\n",
    "    \n",
    "    colors = sns.color_palette('Set2', len(models))\n",
    "    \n",
    "    for b_idx, block in enumerate(blocks):\n",
    "        ax = axes[b_idx]\n",
    "        block_data = sub[sub['block'] == block]\n",
    "        \n",
    "        x = np.arange(len(horizons))\n",
    "        width = 0.13\n",
    "        \n",
    "        for m_idx, model in enumerate(models):\n",
    "            vals = block_data[block_data['model'] == model]['value'].values\n",
    "            offset = (m_idx - len(models)/2 + 0.5) * width\n",
    "            bars = ax.bar(x + offset, vals, width, label=model, color=colors[m_idx], edgecolor='white', linewidth=0.5)\n",
    "            \n",
    "            # Đánh dấu giá trị tốt nhất ở mỗi horizon\n",
    "            for h_idx in range(len(horizons)):\n",
    "                h_vals = block_data[block_data['horizon'] == horizons[h_idx]]['value'].values\n",
    "                best = min(h_vals) if metric_better[metric] == 'lower' else max(h_vals)\n",
    "                if vals[h_idx] == best:\n",
    "                    ax.annotate('★', (x[h_idx] + offset, vals[h_idx]),\n",
    "                               ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')\n",
    "        \n",
    "        ax.set_title(f'{block.upper()}', fontsize=14, fontweight='bold')\n",
    "        ax.set_xticks(x)\n",
    "        ax.set_xticklabels(horizons)\n",
    "        ax.set_xlabel('Horizon')\n",
    "        if b_idx == 0:\n",
    "            ax.set_ylabel(metric)\n",
    "        ax.grid(axis='y', linestyle='--', alpha=0.5)\n",
    "        if b_idx == len(blocks) - 1:\n",
    "            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "    \n",
    "    plt.suptitle(f'Grouped Bar — {metric} (Aggregated) | ★ = Best', fontsize=16, fontweight='bold', y=1.02)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
]))

cells.append(md([
    "**📝 Cách đọc Grouped Bar:**\n",
    "- Ngôi sao ★ đỏ = Model tốt nhất tại Horizon đó.\n",
    "- Nếu 1 model liên tục có ★ ở mọi Horizon, mọi Block → Đó là lựa chọn tối ưu.\n",
    "- Nếu ★ chia đều giữa 2 model → Cân nhắc Hybrid hoặc chọn theo use case (ngắn hạn vs dài hạn)."
]))

# ============================================================
# PART 3: RADAR CHART
# ============================================================
cells.append(md([
    "---\n",
    "## 3️⃣ RADAR CHART — Sức mạnh tổng thể\n",
    "Trung bình R² qua 3 Blocks tại mỗi Horizon. Model nào phình to nhất = tốt nhất toàn diện."
]))

cells.append(code([
    "# Tính trung bình R2 qua tất cả Blocks cho mỗi model\n",
    "r2_agg = df[(df['region'] == 'aggregated') & (df['metric'] == 'R2')]\n",
    "r2_mean = r2_agg.groupby(['model', 'horizon'])['value'].mean().reset_index()\n",
    "\n",
    "fig, ax = plt.subplots(1, 1, figsize=(9, 9), subplot_kw=dict(polar=True))\n",
    "\n",
    "categories = horizons\n",
    "N = len(categories)\n",
    "angles = [n / float(N) * 2 * pi for n in range(N)]\n",
    "angles += angles[:1]  # close the polygon\n",
    "\n",
    "colors = sns.color_palette('tab10', len(models))\n",
    "\n",
    "for i, model in enumerate(models):\n",
    "    values = []\n",
    "    for h in horizons:\n",
    "        v = r2_mean[(r2_mean['model'] == model) & (r2_mean['horizon'] == h)]['value'].values\n",
    "        values.append(v[0] if len(v) > 0 else 0)\n",
    "    values += values[:1]\n",
    "    \n",
    "    ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=colors[i], markersize=6)\n",
    "    ax.fill(angles, values, alpha=0.08, color=colors[i])\n",
    "\n",
    "ax.set_xticks(angles[:-1])\n",
    "ax.set_xticklabels(categories, fontsize=13)\n",
    "ax.set_ylim(0, 100)\n",
    "ax.set_title('Radar Chart — R² Trung Bình qua 3 Blocks (Aggregated)', fontsize=15, fontweight='bold', pad=20)\n",
    "ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md([
    "**📝 Cách đọc Radar:**\n",
    "- Diện tích đa giác càng lớn → Model càng tốt toàn diện.\n",
    "- Nếu đa giác bị \"lõm\" sâu ở 1 đỉnh (ví dụ T+24) → Model yếu ở dải dự báo dài.\n",
    "- XGBoost thường có hình tròn đều nhất (robustness), iTransformer thường nhọn ở T+1 nhưng tụt nhanh."
]))

# ============================================================
# PART 4: BOX PLOT STABILITY
# ============================================================
cells.append(md([
    "---\n",
    "## 4️⃣ BOX PLOT — Đo độ ổn định xuyên Block\n",
    "Gom **tất cả** giá trị của 1 Model (qua 3 Blocks × 3 Regions × 5 Horizons) lại thành 1 Box.\n",
    "- Box **ngắn** (IQR nhỏ) = Model **ổn định**, ít bị ảnh hưởng bởi data shift.\n",
    "- Box **dài** = Model **nhạy cảm**, performance dao động mạnh."
]))

cells.append(code([
    "fig, axes = plt.subplots(1, 3, figsize=(24, 7))\n",
    "\n",
    "palette = sns.color_palette('Set2', len(models))\n",
    "\n",
    "for idx, metric in enumerate(metrics):\n",
    "    ax = axes[idx]\n",
    "    sub = df[df['metric'] == metric]\n",
    "    \n",
    "    sns.boxplot(data=sub, x='model', y='value', ax=ax, palette=palette,\n",
    "                width=0.6, fliersize=3, linewidth=1.5)\n",
    "    \n",
    "    # Overlay stripplot để thấy rõ phân bố\n",
    "    sns.stripplot(data=sub, x='model', y='value', ax=ax,\n",
    "                  color='black', alpha=0.25, size=3, jitter=True)\n",
    "    \n",
    "    ax.set_title(f'Phân bố {metric} (Tất cả Blocks + Regions)', fontsize=14, fontweight='bold')\n",
    "    ax.set_xlabel('')\n",
    "    ax.set_ylabel(metric)\n",
    "    ax.tick_params(axis='x', rotation=25)\n",
    "    ax.grid(axis='y', linestyle='--', alpha=0.5)\n",
    "\n",
    "plt.suptitle('BOX PLOT STABILITY — Model nào ổn định nhất?', fontsize=17, fontweight='bold', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.show()"
]))

cells.append(md([
    "**📝 Cách đọc Box Plot:**\n",
    "- **Median (vạch nằm ngang giữa box):** Giá trị trung vị của model qua tất cả điều kiện.\n",
    "- **Chiều cao Box (IQR):** Phạm vi dao động. Càng ngắn = Model càng consistent.\n",
    "- **Whiskers & Outliers:** Cho thấy trường hợp cực đoan (model bị sụp ở 1 block/region cụ thể).\n",
    "- Với R²: Median cao + Box ngắn = Lý tưởng. Với MAE/RMSE: Median thấp + Box ngắn = Lý tưởng."
]))

# ============================================================
# PART 5: LINE PLOT CHI TIẾT
# ============================================================
cells.append(md([
    "---\n",
    "## 5️⃣ LINE PLOT CHI TIẾT — Phân tích sâu Bắc/Nam/TB từng Block\n",
    "Mỗi Block = 1 figure khổng lồ gồm 3 hàng (RMSE, MAE, R²) × 3 cột (Bắc, Nam, TB)."
]))

cells.append(code([
    "markers = ['o', 's', '^', 'D', 'v', 'p', '*']\n",
    "colors = sns.color_palette('tab10', len(models))\n",
    "\n",
    "for block in blocks:\n",
    "    fig, axes = plt.subplots(3, 3, figsize=(22, 18))\n",
    "    fig.suptitle(f'CHI TIẾT HIỆU NĂNG — {block.upper()}', fontsize=18, fontweight='bold', y=1.01)\n",
    "    \n",
    "    for row, metric in enumerate(metrics):\n",
    "        for col, region in enumerate(regions):\n",
    "            ax = axes[row][col]\n",
    "            for i, model in enumerate(models):\n",
    "                y = raw['data'][block][region][metric][model]\n",
    "                ax.plot(horizons, y, marker=markers[i % len(markers)],\n",
    "                        linewidth=2, label=model, color=colors[i], markersize=7)\n",
    "            \n",
    "            if row == 0:\n",
    "                ax.set_title(f'{region_vn[region]}', fontsize=14, fontweight='bold')\n",
    "            if col == 0:\n",
    "                ax.set_ylabel(metric, fontsize=13, fontweight='bold')\n",
    "            if row == 2:\n",
    "                ax.set_xlabel('Horizon', fontsize=12)\n",
    "            \n",
    "            ax.grid(True, linestyle='--', alpha=0.6)\n",
    "            \n",
    "            if row == 0 and col == 2:\n",
    "                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 10})\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()"
]))

cells.append(md([
    "**📝 Cách đọc Line Plot Chi Tiết:**\n",
    "- Hàng 1 (RMSE): Đường nào thấp nhất = Ít bị sai số cực trị nhất.\n",
    "- Hàng 2 (MAE): Đường nào thấp nhất = Sai số trung bình tuyệt đối nhỏ nhất.\n",
    "- Hàng 3 (R²): Đường nào cao nhất = Bám trend tốt nhất.\n",
    "- So sánh cột Bắc vs cột Nam: Nếu model tốt ở cả 2 → Model robust. Nếu chỉ tốt ở Nam → Model chưa đủ mạnh trong môi trường khó."
]))

# ============================================================
# FINAL SUMMARY
# ============================================================
cells.append(md([
    "---\n",
    "## 🏆 BẢNG TỔNG KẾT ĐỀ XUẤT LỰA CHỌN MÔ HÌNH\n\n",
    "| Tiêu chí | Model đề xuất | Lý do |\n",
    "|---|---|---|\n",
    "| Tốt nhất tổng thể | **XGBoost** | R² cao + ổn định nhất xuyên block |\n",
    "| Tốt nhất ngắn hạn (T+1) | **iTransformer** | MAE thấp nhất nhờ cross-variate attention |\n",
    "| Ổn định nhất | **XGBoost** | Box plot ngắn nhất, ít variance |\n",
    "| Tiềm năng cải thiện | **iTransformer** | Nếu bổ sung feature engineering sẽ rất mạnh |\n",
    "| Deep Learning tốt nhất | **ST-XLinear v7** hoặc **iTransformer** | Tùy block, cạnh tranh sát nhau |\n\n",
    "> 💡 **Gợi ý:** Nếu bạn cần 1 model duy nhất → Chọn **XGBoost**. Nếu bạn muốn thử Hybrid → Kết hợp XGBoost (dài hạn) + iTransformer (ngắn hạn)."
]))

# ============================================================
# BUILD NOTEBOOK
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

import os
os.makedirs("e:/University/Year 3 -2/DA2/CODE/analysis", exist_ok=True)

with open("e:/University/Year 3 -2/DA2/CODE/analysis/benchmark_visualization.ipynb", "w", encoding='utf8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Super Notebook v3 generated successfully!")
