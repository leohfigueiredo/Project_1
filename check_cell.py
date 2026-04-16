import json

notebook_path = r'c:\Project_1\Tensorflow\Files_and_notebooks\08_introduction_to_nlp_in_tensorflow\08_introduction_to_nlp_in_tensorflow.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
print(f'Total cells: {len(cells)}')

if len(cells) >= 122:
    cell_idx = 121
    print(f'Cell 122 type: {cells[cell_idx].get("cell_type")}')
    print(f'Cell 122 source:')
    print(''.join(cells[cell_idx].get('source', [])))
else:
    print('Cell 122 does not exist.')
