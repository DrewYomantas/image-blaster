import hashlib,json,shutil
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
base=root/'.image-blaster/benchmark/synthetic-room-v1'
out=root/'.image-blaster/structure-v1/input'
out.mkdir(parents=True,exist_ok=True)
receipt=json.loads((root/'.image-blaster/benchmark/moge2-vits-normal-v1/M1-receipt.json').read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert digest(receipt['prediction'])==receipt['predictionSha256']
assert receipt['worker']['conditioning']=={'mode':'none'}
assert receipt['worker']['parameters']['independentMonocularViews'] is True
images=[]
for i in range(3):
 name=f'view-{i+1:02}.png'
 src=base/'inputs'/name
 assert digest(src)==receipt['worker']['inputs'][i]['sha256']
 shutil.copyfile(src,out/name)
 images.append({'file':name,'sha256':digest(src),'sourceIndex':i})
with np.load(receipt['prediction'],allow_pickle=False) as src:
 np.savez_compressed(out/'hints.npz',**{k:src[k][:3] for k in ['normals','masks','intrinsics','depths','camera_points']})
manifest={'version':1,'images':images,'neural':{'file':'hints.npz','sha256':digest(out/'hints.npz'),'parentArtifactSha256':receipt['predictionSha256'],'selectedIndices':[0,1,2],'independentMonocularViews':True,'conditioning':{'mode':'none'},'identity':receipt['worker']['identity']},'constraint':{'width_m':1.20,'datum':'horizontal opening mouth between its left and right jambs in the fireplace wall plane; image endpoints separately selected, no evaluation anchor pixels'},'license':{'id':'MIT','commercialUse':'allowed','attribution':'Image Blaster synthetic fixture'}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
files=list((root/'docs/foundation/geometry-results').rglob('*'))+[base/'ground-truth/geometry.json',base/'ground-truth/truth.npz',Path(receipt['prediction'])]+list((base/'inputs').glob('*.png'))
preserved={str(p.relative_to(root)):digest(p) for p in files if p.is_file()}
ledger=root/'.image-blaster/structure-v1/preserved.json'
if ledger.exists():
 for relative,expected in json.loads(ledger.read_text()).items():
  assert digest(root/relative)==expected,'Original evidence changed: '+relative
else:
 ledger.write_text(json.dumps(preserved,indent=2)+'\n')
for source,target in [('annotations-original-v1.json','annotations.json'),('endpoints-original-v1.json','endpoints.json'),('constraint-v1.json','constraint.json')]:
 shutil.copyfile(root/'docs/foundation/geometry-results/structure-v1'/source,out/target)
print(json.dumps({'staged':images,'neuralSha256':manifest['neural']['sha256'],'preservedFiles':len(preserved)},indent=2))
