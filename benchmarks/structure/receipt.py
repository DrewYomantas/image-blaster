import hashlib,json,shutil
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
output=root/'docs/foundation/geometry-results/structure-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt={'version':1,'protocolCommit':'c38a098','startingHead':'ce8dc743dfc4d4d3245d14bc227bb28be9ae6ff8','kind':'structurally fitted, image-assisted where declared; experimental visual-only','decision':3,'decisionText':'Stop automatic one-width room reconstruction; use measured parametric authoring. Local assisted feature sketches remain experimental.','frozenImplementationSha256':sha(root/'workers/structure/fit.py'),'preRunFreeze':json.loads((root/'.image-blaster/structure-v1/freeze.json').read_text(encoding='utf-8-sig')),'datasets':{}}
for dataset in ['structure-v1','structure-v1-supplemental']:
 base=root/'.image-blaster'/dataset
 annotation=json.loads((base/'input/annotations.json').read_text())
 data={'annotation':{'count':annotation['count'],'byView':{v['id']:len(v['points']) for v in annotation['views']},'sha256':sha(base/'input/annotations.json'),'method':annotation['method'],'timeSpentSecondsApprox':annotation.get('timeSpentSecondsApprox'),'revision':annotation['revision']},'inputManifest':json.loads((base/'input/manifest.json').read_text()),'lanes':{}}
 for lane in ['automatic','assisted','ablation']:
  fit=json.loads((base/lane/'receipt.json').read_text());s=json.loads((base/lane/'structure.json').read_text());d=json.loads((base/lane/'diagnostics.json').read_text());e=json.loads((base/'evaluation'/lane/'report-v1.1.json').read_text())
  p=[x['error_m'] for x in e['points'] if x.get('status')=='evaluated']
  stable={x['name']:{'candidateValue':x['value'],'identifiable':x['identifiable'],'range':x['runRange'],'span':x['runRange'][1]-x['runRange'][0]} for x in d['parameters']}
  data['lanes'][lane]={'key':fit['cacheKey'],'status':s['status'],'inputPointCount':sum(len(v['points']) for v in fit['effective']['annotations']['views']),'physicalPolygons':e['geometry_inspection']['polygons'],'rank':d['rank'],'bounds':d['boundContacts'],'trainingReprojection':{'rmsePixels':d['rmsePixels'],'p95Pixels':d['p95Pixels']},'runtime':d['runtime'],'normalSupport':d['normalSupport'],'normalAngleMean':d['normalAngularMeanDegrees'],'numericalGatePassed':d['numericalGatePassed'],'unresolved':s['unknowns'],'canonicalGeometryParameters':s['parameters'],'stability':stable,'widthPerturbations':d['widthPerturbationRuns'],'pointErrors':{'median_m':float(np.median(p)) if p else None,'p95_m':float(np.percentile(p,95)) if p else None,'scope':'named inferred points only; all truth surfaces scored separately'},'evaluation':e,'artifactSha256':{name:sha(base/lane/name) for name in ['receipt.json','structure.json','geometry.obj','diagnostics.json','scene-spec.json','projection-receipt.json']},'readGuardDeniedProbes':d['deniedReadProbes']}
 data['heldoutPolicy']='view4 never fitted or annotated; original is development fixture, supplemental generated independently after freeze; oracle projection is diagnostic only'
 receipt['datasets'][dataset]=data
 for name in ['annotations.json','endpoints.json']:
  if dataset.endswith('supplemental'):shutil.copyfile(base/'input'/name,output/(name.replace('.json','-supplemental-v1.json')))
 shutil.copyfile(base/'evaluation/assisted/overlay-04.png',output/(dataset+'-heldout.png'))
receipt['verification']=json.loads((root/'.image-blaster/structure-v1/verification.json').read_text())
receipt['corrections']=[{'stage':'after freeze, before first complete score','component':'evaluator only','issue':'zero eroded thin-face sample populations produced NaN; changed to all visible truth-face samples, as the all-truth policy requires','fitterChanged':False,'firstResultsPreserved':True},{'stage':'after fits','component':'SceneSpec adapter only','issue':'projection identity now includes canonical scene digest to invalidate metadata/license changes','fitterChanged':False}]
(output/'result-v1.json').write_text(json.dumps(receipt,separators=(',',':'),allow_nan=False)+'\n')
review={'reviewer':'independent Codex adversarial reviewer, separate agent','result':'no remaining actionable defects identified within bounded prototype scope','scopes':['truth isolation','identifiability','assistance fairness','heldout coverage','artifact geometry and overlays','source and cache provenance','final implementation diff'],'resolved':['constraint source mismatch','canonical source metadata cache identity','physically isolated automatic endpoints','aperture-filling wall omitted','unsupported dimension-axis zero estimates','uncertainty visibility','worker source/hint hash verification'],'inspection':['original assisted/ablation OBJ','original assisted/automatic view4 overlays','original ablation view1 overlay','supplemental assisted view4 and ablation view2 overlays'],'fitterSha256':receipt['frozenImplementationSha256'],'recommendation':3,'limitations':['real-photo usefulness unverified','no independent heldout camera solve','no room wall/floor geometry']}
(output/'review-v1.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'receiptBytes':(output/'result-v1.json').stat().st_size,'pointErrors':{ds:{lane:r['pointErrors'] for lane,r in data['lanes'].items()} for ds,data in receipt['datasets'].items()}},indent=2))
