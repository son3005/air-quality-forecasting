import json
import base64
import os

model_path = r'models/wemake STGCN_EVT-GPD_XLinear/model.py'
main_path  = r'models/wemake STGCN_EVT-GPD_XLinear/main.py'
prep_path  = r'preprocess_knowair.py'

with open(model_path, 'r', encoding='utf-8') as f:
    model_code = f.read()
    
with open(main_path, 'r', encoding='utf-8') as f:
    main_code = f.read()
    
with open(prep_path, 'r', encoding='utf-8') as f:
    prep_code = f.read()

# Make the python code work in colab
main_colab = main_code.replace(
    'if __name__ == \'__main__\':', 
    'class Args:\n'
    '    pass\n'
    'args = Args()\n'
    'args.seq_len = 72\n'
    'args.pre_len = 24\n'
    'args.batch_size = 16\n'
    'args.epochs = 100\n'
    'args.lr = 0.001\n'
    'args.target_idx = 0\n'
    'args.limit_years = 2\n'
    'args.threshold = 50.0\n'
    'args.warmup = 15\n'
    'args.patience = 20\n'
    'args.beta1 = 0.99\n'
    'args.beta2 = 0.01\n'
    'args.d_model = 64\n'
    'args.t_ff = 128\n'
    'args.num_nodes = 228\n'
    'args.in_dim = 10\n'
    '\n'
    'import os\n'
    'os.makedirs("data/clean", exist_ok=True)\n'
    '\n'
    'print("Starting Colab Training...")\n'
    '# train(args) # Uncomment to train\n'
    'if False:'
).replace(
    'root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")',
    'root = "data"'
).replace(
    'from model import STGCN_XLinear, EVTGPDLoss',
    '# Model components are defined in the cell above'
)

prep_colab = prep_code.replace(
    'DATA_DIR = "data/KnowAir-V2"',
    'DATA_DIR = "data"\nOUT_DIR = "data/clean"'
)

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 STGCN + EVT-GPD + XLinear (KnowAir-V2 BTHSA)\n",
                "Chạy trực tiếp trên Google Colab. Chọn Runtime -> T4 GPU trước."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install xarray netcdf4 pandas numpy torch scipy tqdm"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Tải Dataset KnowAir-V2 \n",
                "Tải trực tiếp từ Zenodo xuống Colab siêu nhanh."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('data', exist_ok=True)\n",
                "!wget -nc -O data/dataset_bthsa.nc https://zenodo.org/records/15614907/files/dataset_bthsa.nc\n",
                "!wget -nc -O data/stations_bthsa.csv https://zenodo.org/records/15614907/files/stations_bthsa.csv\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Tiền xử lý & Adjacency Matrix"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in prep_colab.split('\n')]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Khởi tạo Model (STGCN_XLinear_GPD)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in model_code.split('\n')]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Bắt đầu Train"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in main_colab.split('\n')] + ['\n\ntrain(args)']
        }
    ],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open('STGCN_KnowAir_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Done!')
