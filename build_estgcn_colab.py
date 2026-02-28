import json
import base64
import os

# Read the STGCN_KnowAir_Colab.ipynb
with open('STGCN_KnowAir_Colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The model cell is index 7
# The train logic cell is index 9

with open('models/wemake STGCN and EVT-GPD/model.py', 'r', encoding='utf-8') as f:
    estgcn_model_code = f.read()

# Make the model code clean for Colab
estgcn_model_code_lines = [line + '\n' for line in estgcn_model_code.split('\n')]
nb['cells'][7]['source'] = estgcn_model_code_lines

# Now we adjust the train logic in cell 9
train_logic = ''.join(nb['cells'][9]['source'])

# 1. Replace the model print
train_logic = train_logic.replace(
    'print("\\n[4/4] Building Hybrid STGCN + XLinear + EVT-GPD v2...")',
    'print("\\n[4/4] Building Baseline E-STGCN (STGCN + EVT-GPD)...")'
)

# 2. Replace model instantiation
train_logic = train_logic.replace(
'''    model = STGCN_XLinear(
        num_nodes=nn_nodes,
        num_features=nf,
        num_timesteps_input=args.seq_len,
        num_timesteps_output=args.pre_len,
        gating_ff=args.t_ff
    ).to(device)''',
'''    model = STGCN(
        num_nodes=nn_nodes,
        num_features=nf,
        num_timesteps_input=args.seq_len,
        num_timesteps_output=args.pre_len
    ).to(device)'''
)

# 3. Replace evaluate print
train_logic = train_logic.replace(
    'print("TESTING — Hybrid STGCN + XLinear + EVT-GPD")',
    'print("TESTING — Baseline E-STGCN (STGCN + EVT-GPD)")'
)

# 4. Replace checkpoints save path
train_logic = train_logic.replace('best_stgcn_xlinear.pth', 'best_estgcn.pth')

# Update title
title_cell = ''.join(nb['cells'][0]['source'])
nb['cells'][0]['source'] = [line + '\n' for line in title_cell.replace('STGCN + EVT-GPD + XLinear', 'Baseline E-STGCN (STGCN + EVT-GPD)').split('\n')[:-1]]

nb['cells'][9]['source'] = [line + '\n' for line in train_logic.split('\n')[:-1]]

with open('ESTGCN_KnowAir_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Built ESTGCN_KnowAir_Colab.ipynb')
