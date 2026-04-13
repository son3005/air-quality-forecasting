"""
train_all.py

Unified Training Script — Chạy toàn bộ mô hình benchmark trên nhiều block
và xuất kết quả ra file JSON giống report/benchmark_metrics.json.

Usage:
    python train_all.py                           # Chạy tất cả models × tất cả blocks
    python train_all.py --blocks block7            # Chạy tất cả models trên block7
    python train_all.py --models XGBoost XLinear   # Chỉ chạy 2 models
    python train_all.py --skip-ensemble            # Bỏ qua Ensemble
"""
import os
import sys
import json
import time
import importlib
import argparse
import numpy as np
from datetime import datetime


def _json_default(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
ALL_BLOCKS = ['block5', 'block7', 'block30']
HORIZONS   = ['T+1', 'T+3', 'T+6', 'T+12', 'T+24']
REGIONS    = ['north', 'south']
METRICS    = ['RMSE', 'MAE', 'R2', 'MAPE']

OUTPUT_PATH = 'report/benchmark_metrics.json'

# Ensemble phải chạy sau XLinear + ESTGCN (vì nó load saved models)
DEFAULT_ORDER = ['XGBoost', 'XLinear', 'iTransformer', 'ESTGCN', 'ST-XLinear', 'Ensemble']


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE MODULE MAPPING
# ══════════════════════════════════════════════════════════════════════════
PIPELINE_MAP = {
    'XGBoost':      ('models.XGBoost.pipeline',      'run_benchmark'),
    'XLinear':      ('models.XLinear.pipeline',       'run_xlinear'),
    'iTransformer': ('models.iTransformer.pipeline',  'run'),
    'ESTGCN':       ('models.ESTGCN.pipeline',        'train_model'),
    'ST-XLinear':   ('models.STXLinear.pipeline',     'run'),
    'Ensemble':     ('models.Ensemble.pipeline',      'run'),
}


# Tất cả thư mục model có thể chứa model.py — cần quản lý sys.path
ALL_MODEL_DIRS = [
    os.path.abspath(os.path.join('models', d))
    for d in ['XGBoost', 'XLinear', 'iTransformer', 'ESTGCN', 'STXLinear', 'Ensemble', 'shared']
]


def set_block_config(module, block_name):
    """Ghi đè BLOCK và DATA_DIR trong module pipeline trước khi chạy."""
    module.BLOCK = block_name
    module.DATA_DIR = f'data/split/{block_name}'


def run_model(model_name, block_name):
    """
    Import (hoặc reload) pipeline module, set block config, rồi chạy.
    Trả về (results, total_time).
    """
    module_path, func_name = PIPELINE_MAP[model_name]

    # Xóa tất cả model dirs cũ khỏi sys.path để tránh import xung đột
    # (VD: iTransformer/model.py bị import thay vì ESTGCN/model.py)
    for d in ALL_MODEL_DIRS:
        if d in sys.path:
            sys.path.remove(d)

    # Thêm đúng thư mục model hiện tại vào đầu sys.path
    model_subdir = module_path.split('.')[1]  # 'XGBoost', 'XLinear', ...
    model_dir = os.path.abspath(os.path.join('models', model_subdir))
    sys.path.insert(0, model_dir)

    # Ensemble cần import từ ESTGCN/ và shared/ → thêm paths phụ
    if model_name == 'Ensemble':
        for extra in ['ESTGCN', 'shared']:
            extra_dir = os.path.abspath(os.path.join('models', extra))
            sys.path.insert(0, extra_dir)

    # Xóa cached 'model' module để tránh import sai
    if 'model' in sys.modules:
        del sys.modules['model']

    # Import hoặc reload pipeline module
    if module_path in sys.modules:
        module = importlib.reload(sys.modules[module_path])
    else:
        module = importlib.import_module(module_path)

    # Ghi đè BLOCK config
    set_block_config(module, block_name)

    # Lấy hàm chạy và thực thi
    run_func = getattr(module, func_name)
    return run_func()


# ══════════════════════════════════════════════════════════════════════════
# AGGREGATE RESULTS → JSON
# ══════════════════════════════════════════════════════════════════════════

def build_benchmark_json(all_block_results, all_training_times):
    """
    Chuyển đổi kết quả sang format benchmark_metrics.json

    Input:
        all_block_results: {block_name: {model_name: [results]}}
        all_training_times: {block_name: {model_name: seconds}}

    Output format:
    {
        "horizons": ["T+1", ...],
        "models": [...],
        "metrics": ["RMSE", "MAE", "R2", "MAPE"],
        "training_times": {block: {model: seconds}},
        "data": {
            "block7": {
                "north": {
                    "RMSE": {"XGBoost": [v1, v2, ...], ...},
                },
                ...
            }
        }
    }
    """
    # Thu thập tất cả model names (giữ thứ tự)
    all_models = []
    for block_results in all_block_results.values():
        for m in block_results:
            if m not in all_models:
                all_models.append(m)

    output = {
        'horizons': HORIZONS,
        'models': all_models,
        'metrics': METRICS,
        'training_times': all_training_times,
        'generated_at': datetime.now().isoformat(),
        'data': {}
    }

    for block_name, model_results in all_block_results.items():
        output['data'][block_name] = {}

        for region in REGIONS:
            region_data = {}

            # 4 metrics chính: RMSE, MAE, R2, MAPE
            for metric in METRICS:
                metric_data = {}
                for model_name, results in model_results.items():
                    values = []
                    for h_str in HORIZONS:
                        matched = [r for r in results
                                   if r['region'] == region and r['horizon'] == h_str]
                        if matched:
                            values.append(round(matched[0][metric], 2))
                        else:
                            values.append(None)
                    metric_data[model_name] = values
                region_data[metric] = metric_data

            # Thời gian train per horizon (seconds)
            train_time_data = {}
            for model_name, results in model_results.items():
                values = []
                for h_str in HORIZONS:
                    matched = [r for r in results
                               if r['region'] == region and r['horizon'] == h_str]
                    if matched and 'train_time' in matched[0]:
                        values.append(matched[0]['train_time'])
                    else:
                        values.append(None)
                train_time_data[model_name] = values
            region_data['train_time'] = train_time_data

            output['data'][block_name][region] = region_data

    return output


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Unified Training Script — Multi-Block')
    parser.add_argument('--models', nargs='+', choices=list(PIPELINE_MAP.keys()),
                        help='Chỉ chạy các mô hình được chỉ định')
    parser.add_argument('--blocks', nargs='+', choices=ALL_BLOCKS, default=ALL_BLOCKS,
                        help=f'Các block cần chạy (default: tất cả {ALL_BLOCKS})')
    parser.add_argument('--skip-ensemble', action='store_true',
                        help='Bỏ qua Ensemble (cần XLinear + ESTGCN đã train)')
    parser.add_argument('--output', default=OUTPUT_PATH,
                        help=f'Đường dẫn file JSON output (default: {OUTPUT_PATH})')
    args = parser.parse_args()

    # Xác định danh sách mô hình cần chạy
    if args.models:
        models_to_run = [m for m in DEFAULT_ORDER if m in args.models]
    else:
        models_to_run = DEFAULT_ORDER.copy()

    if args.skip_ensemble and 'Ensemble' in models_to_run:
        models_to_run.remove('Ensemble')

    blocks_to_run = args.blocks

    print("=" * 70)
    print("  UNIFIED TRAINING PIPELINE — Multi-Block")
    print(f"  Blocks : {blocks_to_run}")
    print(f"  Models : {models_to_run}")
    print(f"  Output : {args.output}")
    print("=" * 70)

    all_block_results = {}   # {block: {model: [results]}}
    all_training_times = {}  # {block: {model: seconds}}
    grand_start = time.time()

    for block_name in blocks_to_run:
        print("\n" + "▓" * 70)
        print(f"  BLOCK: {block_name}  ({blocks_to_run.index(block_name)+1}/{len(blocks_to_run)})")
        print("▓" * 70)

        block_results = {}
        block_times = {}

        for model_name in models_to_run:
            print(f"\n  ┌─ {model_name} × {block_name}")
            print(f"  │")

            try:
                results, total_time = run_model(model_name, block_name)
                block_results[model_name] = results
                block_times[model_name] = round(total_time, 2)
                print(f"  │")
                print(f"  └─ ✅ {model_name} completed in {total_time:.1f}s ({total_time/60:.1f}min)")
            except Exception as e:
                print(f"  │")
                print(f"  └─ ❌ {model_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                block_results[model_name] = []
                block_times[model_name] = 0.0

        all_block_results[block_name] = block_results
        all_training_times[block_name] = block_times

        # ══════════════════════════════════════════════════════════════
        # LƯU SAU MỖI BLOCK (incremental save)
        # ══════════════════════════════════════════════════════════════
        benchmark = build_benchmark_json(all_block_results, all_training_times)
        benchmark['total_training_time'] = round(time.time() - grand_start, 2)

        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(benchmark, f, indent=2, ensure_ascii=False, default=_json_default)

        print(f"\n  💾 Saved {block_name} results → {args.output}")

    grand_total = time.time() - grand_start

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Grand total: {grand_total:.1f}s ({grand_total/60:.1f}min)")
    print(f"  Results saved to: {args.output}")

    for block_name in blocks_to_run:
        print(f"\n  [{block_name}]")
        for name, t in all_training_times.get(block_name, {}).items():
            status = "✅" if t > 0 else "❌"
            print(f"    {status} {name:<15s} {t:>8.1f}s ({t/60:.1f}min)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
