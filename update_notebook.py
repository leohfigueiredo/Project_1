import json

notebook_path = r'c:\Project_1\Tensorflow\Files_and_notebooks\08_introduction_to_nlp_in_tensorflow\08_introduction_to_nlp_in_tensorflow.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find cell with execution_count 122 or just use index 122 as fallback
target_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code' and cell.get('execution_count') == 122:
        target_idx = i
        break

if target_idx is None:
    target_idx = 122

code_to_add = [
    "\n\n# --- Adicionado conforme solicitado ---\n",
    "# 1. Identificando os dois melhores modelos:\n",
    "best_models = all_model_results.sort_values(by='f1', ascending=False).head(2)\n",
    "print('Os dois melhores modelos são:')\n",
    "print(best_models)\n",
    "\n",
    "# 2. Setando o TensorBoard para a pasta de logs dos modelos\n",
    "%load_ext tensorboard\n",
    "%tensorboard --logdir \"model_logs\" --port 6010\n"
]

nb['cells'][target_idx]['source'].extend(code_to_add)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated cell at index {target_idx} successfully.")
