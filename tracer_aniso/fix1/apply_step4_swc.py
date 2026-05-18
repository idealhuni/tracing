"""
step4_viz.ipynb — 두 가지 수정:

1. Labeled SWC 저장 셀 추가 (cell 12 뒤에 삽입)
   - Axon → type 2, Apical → type 4, Basal → type 3, Soma → type 1
   - 모든 자손 노드에 primary branch 분류 전파

2. Vaa3D 검정 구 제거
   - 원인: type=1 soma 노드의 radius=10.77 µm → Vaa3D가 채워진 구로 렌더링
   - 수정: labeled SWC에서 soma 노드 radius를 1.0 µm로 축소
           (실제 소마 크기는 soma.npz에 보존)

대상: fix1/step4_viz.ipynb  (원본 미변경)
"""
import json, pathlib

BASE = pathlib.Path(__file__).parent

path = BASE / 'step4_viz.ipynb'
with open(path) as f:
    nb = json.load(f)

NEW_CELL_SOURCE = '''\
# ── Save labeled SWC (Axon / Apical / Basal) ─────────────────
# SWC type codes used:
#   1 = soma
#   2 = axon
#   3 = basal dendrite
#   4 = apical dendrite
#
# Vaa3D black-sphere fix:
#   The original SWC stores soma radius = soma_r_um (~10 µm).
#   Vaa3D renders type-1 nodes as a filled sphere with that radius,
#   producing the black overlapping ball. Setting display radius to
#   1.0 µm suppresses the sphere while keeping the tree structure.

SOMA_RADIUS_DISPLAY = 1.0   # µm — display only; actual stored in soma.npz
LABEL_TO_TYPE = {'Axon': 2, 'Apical': 4, 'Basal': 3}

# Build primary-ancestor lookup (child-of-soma → label)
primary_label = {pid: labels[pid] for pid in pids}

def node_swc_type(nid):
    """Walk up to primary ancestor and return its SWC type."""
    if nid == 1:
        return 1
    cur = nid
    while cur in swc_nodes and swc_nodes[cur]['parent'] not in (-1, 1):
        cur = swc_nodes[cur]['parent']
    lbl = primary_label.get(cur, 'Basal')
    return LABEL_TO_TYPE[lbl]

out_lines = [
    f'# tracer_aniso LABELED — Axon(type2) / Apical(type4) / Basal(type3)',
    f'# axon_pid={axon_pid}  apical_pid={apical_pid}',
    f'# soma_r_actual={soma_r_um:.2f} µm  soma_r_display={SOMA_RADIUS_DISPLAY} µm',
    f'# Vaa3D: soma display radius reduced to suppress rendered sphere',
    '# id type x y z radius parent',
]

type_counts = {}
for nid in sorted(swc_nodes.keys()):
    n = swc_nodes[nid]
    t = node_swc_type(nid)
    r = SOMA_RADIUS_DISPLAY if nid == 1 else n['r']
    out_lines.append(
        f"{nid} {t} {n['x']:.4f} {n['y']:.4f} {n['z']:.4f} {r:.4f} {n['parent']}"
    )
    type_counts[t] = type_counts.get(t, 0) + 1

OUT_LABELED = SWC_FILE.replace('neurons_auto.swc', 'neurons_labeled.swc')
with open(OUT_LABELED, 'w') as f:
    f.write('\\n'.join(out_lines) + '\\n')

TYPE_NAMES = {1: 'soma', 2: 'axon', 3: 'basal', 4: 'apical'}
print(f'Saved: {OUT_LABELED}')
print(f'Node type distribution:')
for t, count in sorted(type_counts.items()):
    print(f'  type {t} ({TYPE_NAMES[t]:7s}): {count:,} nodes')
print(f'\\nVaa3D fix: soma radius {soma_r_um:.2f} µm → {SOMA_RADIUS_DISPLAY} µm (display only)')
'''

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "save_labeled_swc",
    "metadata": {},
    "outputs": [],
    "source": NEW_CELL_SOURCE.splitlines(keepends=True),
}

# cell 12 (a56aa29e, classification) 바로 뒤에 삽입
insert_after = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'axon_pid = min(pids' in src and "labels[p] = 'Axon'" in src:
        insert_after = i
        break

if insert_after is None:
    print('ERROR: classification cell not found')
    exit(1)

nb['cells'].insert(insert_after + 1, new_cell)
print(f'New cell inserted after cell {insert_after} (classification)')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# 검증
with open(path) as f:
    text = f.read()

checks = [
    ('SOMA_RADIUS_DISPLAY', 'soma display radius param'),
    ('neurons_labeled.swc', 'labeled SWC filename'),
    ('node_swc_type',       'type assignment function'),
    ('Vaa3D',               'Vaa3D explanation comment'),
    ('LABEL_TO_TYPE',       'label→type mapping'),
]
print()
for kw, label in checks:
    status = '✓' if kw in text else '✗ MISSING'
    print(f'  {status}  {label}')

# 셀 수 확인
print(f'\nTotal cells: {len(nb["cells"])} (was 14, now 15)')
