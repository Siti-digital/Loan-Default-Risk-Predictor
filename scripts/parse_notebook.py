import json

with open("macro_stress_testing.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = ''.join(cell['source']).strip()
    first_line = source.split('\n')[0] if source else ''
    outputs = cell.get('outputs', [])
    out_types = [o.get('output_type', '') for o in outputs]
    print(f"Cell {i:02d} [{ctype}]: {first_line[:75]} | outputs: {len(outputs)} {out_types}")
