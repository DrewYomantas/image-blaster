import hashlib,json,shutil,subprocess,sys,tempfile,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
base=root/'.image-blaster/structure-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command(args,success=True):
 p=subprocess.run(args,cwd=root,text=True,capture_output=True)
 if success and p.returncode: raise RuntimeError(p.stderr+p.stdout)
 if not success and not p.returncode: raise RuntimeError('Expected refusal: '+str(args))
 return p

def fit(input,output,lane='assisted'):
 p=command(['node','engine/cli.mjs','structure','--input',str(input),'--out',str(output),'--lane',lane])
 value=json.loads(p.stdout); value['worker']=json.loads(value['worker']); return value

results={'replay':{},'exports':{},'boundary':{},'preserved':{}}
for dataset in ['structure-v1','structure-v1-supplemental']:
 folder=root/'.image-blaster'/dataset
 for lane in ['automatic','assisted','ablation']:
  result=fit(folder/'input',folder/lane,lane)
  assert result['worker']['cached'] is True
  result=fit(folder/'input',folder/lane,lane)
  assert result['worker']['cached'] is True and result['projectionCached'] is True
  results['replay'][dataset+'/'+lane]=result
  command(['node','engine/cli.mjs','validate','--scene',result['scenePath']])
  command(['node','engine/cli.mjs','export','--scene',result['scenePath'],'--target','benson','--out',str(folder/lane/'visual-export.json')])
  blocked=command(['node','engine/cli.mjs','export','--scene',result['scenePath'],'--target','benson','--purpose','technical','--out',str(folder/lane/'must-not-exist.json')],False)
  assert not (folder/lane/'must-not-exist.json').exists()
  results['exports'][dataset+'/'+lane]={'reopened':True,'visualExport':True,'technicalBlocked':blocked.stderr.strip()}
scratch=Path(tempfile.mkdtemp(prefix='verification-',dir=base))
input=scratch/'input';shutil.copytree(base/'input',input)
(input/'evaluation-metadata.json').write_text(json.dumps({'heldout':'evaluation-only sentinel, must not enter identity'}))
replay=json.loads(command([sys.executable,'workers/structure/fit.py','--input',str(input),'--output',str(base/'assisted'),'--lane','assisted']).stdout)
assert replay['cached']
results['boundary']['evaluationMetadataExcluded']=True
constraint=json.loads((input/'constraint.json').read_text(encoding='utf-8-sig'));constraint['width_m']*=1.01
(input/'constraint.json').write_text(json.dumps(constraint))
changed=fit(input,scratch/'width-change')
assert not changed['worker']['cached']
assert changed['worker']['cacheKey']!=results['replay']['structure-v1/assisted']['worker']['cacheKey']
scene=json.loads(Path(changed['scenePath']).read_text())
assert scene['objects'][0]['dimensions']['width'][0]['value']==1.212
results['boundary']['widthChangedKey']=changed['worker']['cacheKey']
shutil.copyfile(base/'input/constraint.json',input/'constraint.json')
a=json.loads((input/'annotations.json').read_text());a['views'][0]['points']['opening.topLeft'][0]+=1
(input/'annotations.json').write_text(json.dumps(a))
changed=fit(input,scratch/'annotation-change')
assert changed['worker']['cacheKey']!=results['replay']['structure-v1/assisted']['worker']['cacheKey']
results['boundary']['annotationChangedKey']=changed['worker']['cacheKey']
# Ablation is independent of the actual hint bytes, including corrupted unused hints.
shutil.copyfile(base/'input/annotations.json',input/'annotations.json')
(input/'hints.npz').write_bytes(b'unused neural bytes')
assert json.loads(command([sys.executable,'workers/structure/fit.py','--input',str(input),'--output',str(base/'ablation'),'--lane','ablation']).stdout)['cached']
results['boundary']['ablationDoesNotReadNeuralBytes']=True
bad=command(['node','engine/cli.mjs','structure','--input',str(input),'--out',str(scratch/'bad-hints'),'--lane','assisted'],False)
assert 'Neural bytes' in bad.stderr
results['boundary']['neuralMismatchRejected']=True
for relative,expected in json.loads((base/'preserved.json').read_text()).items():
 assert sha(root/relative)==expected,relative
results['preserved']={'files':len(json.loads((base/'preserved.json').read_text())),'allUnchanged':True}
results['workerImplementationSha256']=sha(root/'workers/structure/fit.py')
results['scratch']=str(scratch)
(base/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({'replayed':len(results['replay']),'exported':len(results['exports']),'boundary':results['boundary'],'preserved':results['preserved']},indent=2))
