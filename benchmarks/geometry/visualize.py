from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

r=Path(__file__).resolve().parents[2]
output=r/'.image-blaster/benchmark/camera-conditioning-v1'
evidence=r/'docs/foundation/geometry-results/camera-conditioning-v1'
sys.path.insert(0,str(r/'benchmarks/geometry'))
from evaluate import load_prediction

rows=[('Source RGB',r/'.image-blaster/benchmark/synthetic-room-v1/inputs','view-'),
      ('Analytic truth, m',output/'lane-A-supplemental','GT-depth-'),
      ('A: opening-anchor aligned, m',output/'lane-A-supplemental','ALIGNED-depth-'),
      ('B: intrinsics control + anchor, m',output/'lane-B','ALIGNED-depth-'),
      ('C: oracle pose native, m',output/'lane-C','NATIVE-depth-'),
      ('D: perturbed pose native, m',output/'lane-D','NATIVE-depth-')]
fig,axes=plt.subplots(len(rows),4,figsize=(14,13))
for row,(label,folder,prefix) in enumerate(rows):
    for view in range(4):
        axes[row,view].imshow(Image.open(folder/f'{prefix}{view+1:02}.png'))
        axes[row,view].axis('off')
        axes[row,view].set_title(f'{label} / view {view+1}',fontsize=9)
fig.suptitle('Frozen synthetic room: identical images, inferred depth; all metric depth colours use fixed 0-6 m range',fontsize=12)
fig.tight_layout()
fig.savefig(evidence/'depth-comparison-v1.png',dpi=110)
plt.close(fig)

rows=[('A: opening-anchor aligned',output/'lane-A-supplemental','ALIGNED'),
      ('B: intrinsics control + anchor',output/'lane-B','ALIGNED'),
      ('C: oracle pose native',output/'lane-C','NATIVE'),
      ('D: perturbed pose native',output/'lane-D','NATIVE')]
fig,axes=plt.subplots(4,4,figsize=(14,9))
for row,(label,folder,prefix) in enumerate(rows):
    for view in range(4):
        source=(view+1)%4
        axes[row,view].imshow(Image.open(folder/f'{prefix}-cross-view-{source+1:02}-to-{view+1:02}.png'))
        axes[row,view].axis('off')
        axes[row,view].set_title(f'{label} / source {source+1} to GT {view+1}',fontsize=9)
fig.suptitle('Cross-view point projections into fixed truth cameras: holes retained; no mesh or surface completion',fontsize=12)
fig.tight_layout()
fig.savefig(evidence/'cross-view-comparison-v1.png',dpi=110)
plt.close(fig)

a=json.loads((r/'docs/foundation/geometry-results/da3-small-cpu-v1.json').read_text())
b=json.loads((evidence/'lane-B-result-v1.json').read_text())
pa,pb=load_prediction(a['metrics']['prediction']),load_prediction(b['metrics']['prediction'])
comparison={name:dict(exact_array_equality=bool(np.array_equal(pa[name],pb[name])),max_absolute_difference=float(np.max(np.abs(pa[name]-pb[name]))),shape=list(pa[name].shape)) for name in ['depths','confidences','intrinsics','extrinsics']}
receipt=dict(kind='negative control exact tensor comparison, no Lane A inference rerun',lane_a_prediction_sha256=a['metrics']['prediction_sha256'],lane_b_prediction_sha256=b['metrics']['prediction_sha256'],arrays=comparison)
(evidence/'intrinsics-negative-control-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
