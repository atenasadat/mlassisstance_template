import json
import argparse
from typing import Dict, List, Literal


parser = argparse.ArgumentParser(description='Append an inventory to deploy workflow.')
parser.add_argument('inventory', type=str, help='inventory name.')

args = parser.parse_args()
with open('.github/conf/matrix.json', 'r') as f:
    matrix: Dict[Literal['inventory'], List[str]] = json.load(f)

if args.inventory not in matrix['inventory']:
    matrix['inventory'].append(args.inventory)

with open('.github/conf/matrix.json', 'w') as f:
    json.dump(matrix, f)
