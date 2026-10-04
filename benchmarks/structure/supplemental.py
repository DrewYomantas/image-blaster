import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'geometry'))
from fixture import WIDTH, HEIGHT, camera, render, png, depth_image

VERSION = 'structural-sanity-room-v1'


def scene():
    values = [
        ('floor',1,[-2.9,-.1,-.5],[2.6,0,4.1],[.53,.37,.23]),
        ('ceiling',2,[-2.9,3.05,-.5],[2.6,3.15,4.1],[.86,.85,.8]),
        ('left-wall',3,[-2.9,0,-.5],[-2.8,3.05,4.1],[.68,.72,.70]),
        ('right-wall',4,[2.5,0,-.5],[2.6,3.05,4.1],[.68,.72,.70]),
        ('rear-wall',5,[-2.9,0,4.0],[2.6,3.05,4.1],[.75,.75,.70]),
        ('front-left',6,[-2.8,0,-.1],[-.6,3.05,0],[.79,.76,.69]),
        ('front-right',6,[.6,0,-.1],[2.5,3.05,0],[.79,.76,.69]),
        ('front-bottom',6,[-.6,0,-.1],[.6,.62,0],[.65,.62,.56]),
        ('front-top',6,[-.6,1.72,-.1],[.6,3.05,0],[.79,.76,.69]),
        ('opening-back',7,[-.6,.62,-.49],[.6,1.72,-.39],[.12,.13,.14]),
        ('opening-left',8,[-.7,.62,-.39],[-.6,1.72,0],[.23,.23,.23]),
        ('opening-right',8,[.6,.62,-.39],[.7,1.72,0],[.23,.23,.23]),
        ('opening-bottom',8,[-.6,.52,-.39],[.6,.62,0],[.23,.23,.23]),
        ('opening-top',8,[-.6,1.72,-.39],[.6,1.82,0],[.23,.23,.23]),
        ('hearth',9,[-1.1,0,0],[1.1,.21,.43],[.37,.38,.38]),
        ('mantel',10,[-1.06,1.94,0],[1.06,2.11,.33],[.28,.16,.08]),
        ('tall-left-box',11,[-2.0,0,.72],[-1.42,1.14,1.26],[.27,.37,.42]),
        ('foreground-stool',12,[-.70,0,2.09],[-.12,.48,2.65],[.58,.31,.18]),
        ('right-seat-box',13,[1.24,0,1.08],[2.02,.92,1.84],[.34,.44,.28]),
    ]
    return [dict(name=n,label=l,low=lo,high=hi,color=c) for n,l,lo,hi,c in values]


def generate(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('preserve first supplemental fixture; output must be empty')
    source, truth = output / 'inputs', output / 'ground-truth'
    source.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    primitives = scene()
    intrinsics = np.array([[[f,0,319.5],[0,f,239.5],[0,0,1]] for f in [375.,360.,390.,365.]])
    positions = [[-1.80,1.72,3.43],[-.28,1.38,3.66],[1.23,1.85,3.38],[1.94,1.58,3.12]]
    extrinsics = np.array([camera(np.array(p),[-.03,1.38,.26]) for p in positions])
    images, depths, labels = zip(*(render(k,e,primitives) for k,e in zip(intrinsics,extrinsics)))
    hashes = []
    for i, rgb in enumerate(images):
        name = f'view-{i+1:02}.png'
        png(source/name,rgb)
        hashes.append(dict(file=name,sha256=hashlib.sha256((source/name).read_bytes()).hexdigest()))
        png(truth/f'SYNTHETIC-input-{i+1:02}.png',rgb,f'SYNTHETIC INPUT {i+1:02}')
        png(truth/f'GT-depth-{i+1:02}.png',depth_image(depths[i]),f'SYNTHETIC GT DEPTH {i+1:02}')
    dimensions = [dict(id='room',labels=[1,2,3,4,5,6],dimensions_m=[5.3,3.05,4.0]),
                  dict(id='opening-back',labels=[7],dimensions_m=[1.2,1.10,0.]),
                  dict(id='opening-recess',labels=[7,8],dimensions_m=[1.2,1.10,.39]),
                  *[dict(id=p['name'],labels=[p['label']],dimensions_m=(np.array(p['high'])-p['low']).tolist()) for p in primitives if p['label']>=9]]
    metadata = dict(version=VERSION,kind='synthetic',renderer='existing numpy-analytic-box-cpu-v1',
                    generator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    renderer_source_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'geometry/fixture.py').read_bytes()).hexdigest(),
                    numpy_version=np.__version__,resolution=[WIDTH,HEIGHT],world_frame='metres, right-handed, x-right y-up z-toward-rear-wall',
                    camera_frame='world-to-camera; x-right y-down z-forward; integer pixel centres',depth='positive camera-z metres',
                    intrinsics=intrinsics.tolist(),extrinsics=extrinsics.tolist(),primitives=primitives,dimensions=dimensions,input_hashes=hashes,
                    experiment=dict(first_fixture=True,no_tuning=True,created_after_parent_freeze_commit='c38a098',heldout_view=4,
                                    sole_metric_input=dict(kind='opening-mouth-width',width_m=1.2),
                                    differences='visible proportions, asymmetric room spans, opening height/elevation, hearth/mantel sizes, furniture arrangement and camera arrangement'))
    (source/'manifest.json').write_text(json.dumps(dict(fixture=VERSION,kind='synthetic',rights='project-authored synthetic fixture; MIT',ordered_images=hashes),indent=2)+'\n')
    np.savez_compressed(truth/'truth.npz',depths=depths,labels=labels,intrinsics=intrinsics,extrinsics=extrinsics)
    (truth/'geometry.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return dict(fixture=VERSION,output=str(output.resolve()),images=hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.out),indent=2))
