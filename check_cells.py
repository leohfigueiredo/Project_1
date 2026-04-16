import json

notebook_path = r'c:\Project_1\Tensorflow\Files_and_notebooks\08_introduction_to_nlp_in_tensorflow\08_introduction_to_nlp_in_tensorflow.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in range(120, 137):
    cell = nb['cells'][i]
    source = ''.join(cell.get('source', []))
    print(f'Cell index {i} type: {cell.get("cell_type")}\n{source[:200]}\n')
