import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import ctypes

import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image

VERSION = 'structure-v1.0'
SETTINGS = dict(seed=1705, max_nfev=120, starts=3, perturbations=2, pixelSigma=2.0,
                normalSigma=math.sin(math.radians(10)), normalWeight=0.5,
                loss='soft_l1', rankRelativeTolerance=1e-6, boundFraction=1e-3,
                focalBounds=[0.35, 3.0], rotationBounds=[-0.9, 0.9], depthTerms=False)
SPECS = {
 'openingHeight': (0.9, .15, 3.5),
 'hearthCenter': (0., -3., 3.), 'hearthWidth': (1.8, .3, 6.),
 'hearthTop': (-.1, -2., 2.), 'hearthThickness': (.15, .02, 1.),
 'hearthBack': (0., -1., 1.), 'hearthDepth': (.5, .03, 3.),
 'mantelCenter': (0., -3., 3.), 'mantelWidth': (2., .3, 6.),
 'mantelBottom': (1.3, .2, 4.), 'mantelThickness': (.15, .02, 1.),
 'mantelBack': (0., -1., 1.), 'mantelDepth': (.3, .03, 2.),
 'roomLeft': (-2., -10., -.61), 'roomRight': (2., .61, 10.),
 'roomFloor': (-.2, -3., 0.), 'roomHeight': (3., 1.5, 8.), 'roomFront': (5., .5, 15.)}
POINT_DEPS = {}
for side in ['Left', 'Right']:
    POINT_DEPS['opening.bottom'+side] = []
    POINT_DEPS['opening.top'+side] = ['openingHeight']
    for level, depth in [('top','Back'), ('top','Front'), ('bottom','Front')]:
        POINT_DEPS['hearth.'+level+depth+side] = ['hearthCenter','hearthWidth','hearthTop','hearthBack'] + (['hearthThickness'] if level=='bottom' else []) + (['hearthDepth'] if depth=='Front' else [])
    for level, depth in [('bottom','Back'), ('bottom','Front'), ('top','Front')]:
        POINT_DEPS['mantel.'+level+depth+side] = ['mantelCenter','mantelWidth','mantelBottom','mantelBack'] + (['mantelThickness'] if level=='top' else []) + (['mantelDepth'] if depth=='Front' else [])
    for depth in ['back','front']:
        for level in ['Floor','Ceiling']:
            POINT_DEPS['room.'+depth+level+side] = ['room'+side,'roomFloor'] + (['roomHeight'] if level=='Ceiling' else []) + (['roomFront'] if depth=='front' else [])
FACES = {
 'opening.mouth':['opening.bottomLeft','opening.bottomRight','opening.topRight','opening.topLeft'],
 'hearth.top':['hearth.topBackLeft','hearth.topBackRight','hearth.topFrontRight','hearth.topFrontLeft'],
 'hearth.front':['hearth.bottomFrontLeft','hearth.bottomFrontRight','hearth.topFrontRight','hearth.topFrontLeft'],
 'mantel.bottom':['mantel.bottomBackLeft','mantel.bottomBackRight','mantel.bottomFrontRight','mantel.bottomFrontLeft'],
 'mantel.front':['mantel.bottomFrontLeft','mantel.bottomFrontRight','mantel.topFrontRight','mantel.topFrontLeft'],
 'room.back':['room.backFloorLeft','room.backFloorRight','room.backCeilingRight','room.backCeilingLeft'],
 'room.floor':['room.backFloorLeft','room.frontFloorLeft','room.frontFloorRight','room.backFloorRight'],
 'room.ceiling':['room.backCeilingLeft','room.backCeilingRight','room.frontCeilingRight','room.frontCeilingLeft'],
 'room.left':['room.backFloorLeft','room.backCeilingLeft','room.frontCeilingLeft','room.frontFloorLeft'],
 'room.right':['room.backFloorRight','room.frontFloorRight','room.frontCeilingRight','room.backCeilingRight']}
ANCHORS = ['opening.bottomLeft','opening.bottomRight']

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def point(name, p):
    side = -1 if name.endswith('Left') else 1
    if name.startswith('opening.'):
        return np.array([side*p.get('_width',1.2)/2, p.get('openingHeight', .9) if '.top' in name else 0., 0.])
    entity, part = name.split('.')
    if entity in ['hearth','mantel']:
        x = p[entity+'Center'] + side*p[entity+'Width']/2
        y = p[entity+('Top' if entity=='hearth' else 'Bottom')]
        if entity=='hearth' and part.startswith('bottom'): y -= p['hearthThickness']
        if entity=='mantel' and part.startswith('top'): y += p['mantelThickness']
        z = p[entity+'Back'] + (p[entity+'Depth'] if 'Front' in part else 0.)
        return np.array([x,y,z])
    return np.array([p['roomLeft' if side<0 else 'roomRight'], p['roomFloor']+(p['roomHeight'] if 'Ceiling' in part else 0.), p['roomFront'] if part.startswith('front') else 0.])

def camera(values, size):
    r = Rotation.from_rotvec(values[:3]).as_matrix() @ np.diag([1.,-1.,-1.])
    c = np.asarray(values[3:6]); f = float(values[6])*max(size)
    k = np.array([[f,0,(size[0]-1)/2],[0,f,(size[1]-1)/2],[0,0,1.]])
    return k, np.column_stack([r, -r@c])

def project(points, k, pose):
    cam = np.asarray(points) @ pose[:,:3].T + pose[:,3]
    uvw = cam @ k.T
    uv = uvw[:,:2]/np.maximum(cam[:,2,None], 1e-4)
    return uv, cam[:,2]

def validate_annotations(data, sizes, automatic=False):
    if data.get('schemaVersion') != 1 or not isinstance(data.get('views'), list): raise ValueError('Unsupported annotations')
    result = {}; count = 0
    for view in data['views']:
        ident = view['id']
        if ident not in sizes or ident in result: raise ValueError('Unknown/duplicate view')
        points = view['points']
        if not isinstance(points, dict) or len(points)>24: raise ValueError('Annotation budget exceeded')
        clean = {}
        for name, xy in points.items():
            if name not in POINT_DEPS: raise ValueError('Unknown semantic: '+name)
            if not isinstance(xy,list) or len(xy)!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in xy): raise ValueError('Invalid pixels')
            if not (0<=xy[0]<sizes[ident][0] and 0<=xy[1]<sizes[ident][1]): raise ValueError('Pixel outside image')
            if automatic and name not in ANCHORS: raise ValueError('Automatic lane must not receive assisted points')
            clean[name]=list(xy)
        if any(a not in clean for a in ANCHORS): raise ValueError('Missing mouth width endpoints')
        if np.linalg.norm(np.array(clean[ANCHORS[1]])-clean[ANCHORS[0]])<5: raise ValueError('Conflicting/degenerate mouth endpoints')
        if clean[ANCHORS[0]][0]>=clean[ANCHORS[1]][0]: raise ValueError('Mouth endpoint ordering conflict')
        count += len(clean); result[ident]=clean
    if len(result)!=3 or count>72: raise ValueError('Exactly three views and <=72 annotations required')
    if 'anchor' in data: raise ValueError('Metric anchors must be supplied only through constraint.json')
    return result

def automatic_points(images, observations):
    evidence = []
    for ident, rgb in images.items():
        pts=observations[ident]; left=np.array(pts[ANCHORS[0]]); right=np.array(pts[ANCHORS[1]])
        width=np.linalg.norm(right-left); direction=(right-left)/width; up=np.array([direction[1],-direction[0]])
        edges=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),60,150)
        lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(12,int(width*.15)),minLineLength=max(10,int(width*.55)),maxLineGap=8)
        candidates=[]
        for raw in ([] if lines is None else lines[:,0,:]):
            a=np.array(raw[:2],float); b=np.array(raw[2:],float)
            if np.dot(b-a,direction)<0: a,b=b,a
            length=np.linalg.norm(b-a)
            if length<1 or abs(np.cross(direction,(b-a)/length))>.12: continue
            height=np.dot((a+b-left-right)/2,up)
            edge_error=(abs(np.dot(a-left,direction))+abs(np.dot(b-right,direction)))/width
            if .2*width<height<1.5*width and edge_error<.4:
                candidates.append((edge_error, -length, a.tolist(), b.tolist()))
        candidates.sort()
        if candidates:
            chosen=candidates[0]; pts['opening.topLeft']=chosen[2]; pts['opening.topRight']=chosen[3]
        evidence.append(dict(view=ident,candidates=len(candidates),selected=(candidates[0][2:] if candidates else None),method='RGB Canny/Hough bounded mouth top proposal',unsupported=['hearth','mantel','room extents']))
    return evidence

def normal_clusters(normals, mask):
    n=np.asarray(normals[::8,::8],float).reshape(-1,3); m=np.asarray(mask[::8,::8]).reshape(-1).astype(bool)
    lengths=np.linalg.norm(n,axis=1); n=n[m & np.isfinite(n).all(axis=1) & (lengths>.5)]
    if not len(n): return np.empty((0,3))
    n=n/np.linalg.norm(n,axis=1)[:,None]
    major=np.argmax(np.abs(n),axis=1); n=n*np.where(n[np.arange(len(n)),major]<0,-1,1)[:,None]
    bins=np.round(n*8).astype(int); unique,inv,counts=np.unique(bins,axis=0,return_inverse=True,return_counts=True)
    selected=[]
    for ix in np.argsort(-counts,kind='stable'):
        mean=n[inv==ix].mean(axis=0); mean/=np.linalg.norm(mean)
        if all(abs(np.dot(mean,s))<math.cos(math.radians(15)) for s in selected): selected.append(mean)
        if len(selected)==6: break
    return np.asarray(selected)

class Problem:
    def __init__(self, observations, sizes, normals, width=1.2):
        self.obs=observations; self.sizes=sizes; self.normals=normals; self.width=width
        scale=width/1.2
        used=set(k for pts in observations.values() for k in pts)
        needed=set(d for k in used for d in POINT_DEPS[k])
        self.names=[k for k in SPECS if k in needed]; self.ids=sorted(observations)
        self.initial=np.array([SPECS[k][0]*scale for k in self.names]+[v for ident in self.ids for v in [0,0,0,0,.5*scale,4*scale,1.]])
        self.lower=np.array([SPECS[k][1]*scale for k in self.names]+[-.9,-.9,-.9,-10*scale,-5*scale,.3*scale,.35]*len(self.ids))
        self.upper=np.array([SPECS[k][2]*scale for k in self.names]+[.9,.9,.9,10*scale,8*scale,15*scale,3.]*len(self.ids))
        self.labels=self.names+[ident+'.'+k for ident in self.ids for k in ['rx','ry','rz','cx','cy','cz','focalRatio']]
    def unpack(self,x):
        p=dict(zip(self.names,x[:len(self.names)])); p['_width']=self.width; cams={}
        for i,ident in enumerate(self.ids): cams[ident]=camera(x[len(self.names)+i*7:len(self.names)+(i+1)*7],self.sizes[ident])
        return p,cams
    def residual(self,x):
        p,cams=self.unpack(x); residual=[]
        for ident in self.ids:
            names=list(self.obs[ident]); xyz=np.array([point(k,p) for k in names]); k,pose=cams[ident]
            uv,z=project(xyz,k,pose)
            residual.extend(((uv-np.array(list(self.obs[ident].values())))/SETTINGS['pixelSigma']).ravel())
            residual.extend(np.minimum(z-.05,0)*100)
            axes=pose[:,:3].T
            for n in self.normals.get(ident,[]):
                axis=axes[np.argmax(np.abs(axes@n))]
                residual.extend(np.cross(n,axis)*SETTINGS['normalWeight']/SETTINGS['normalSigma'])
        return np.asarray(residual)
    def fit(self, initial=None):
        return least_squares(self.residual, self.initial if initial is None else initial, bounds=(self.lower,self.upper),loss=SETTINGS['loss'],f_scale=1,max_nfev=SETTINGS['max_nfev'],x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8)

def finite_jacobian(problem,x):
    base=problem.residual(x); cols=[]
    for i in range(len(x)):
        h=1e-5*max(1,abs(x[i])); plus=x.copy(); minus=x.copy(); plus[i]+=h; minus[i]-=h
        cols.append((problem.residual(plus)-problem.residual(minus))/(2*h))
    return np.column_stack(cols)

def rank_report(jac):
    singular=np.linalg.svd(jac,compute_uv=False)
    rank=int(np.count_nonzero(singular>(singular[0]*SETTINGS['rankRelativeTolerance'] if len(singular) else 0)))
    return dict(rank=rank,columns=jac.shape[1],rows=jac.shape[0],nullity=jac.shape[1]-rank,singularValues=singular.tolist())

def solve(observations,sizes,normals,width=1.2):
    problem=Problem(observations,sizes,normals,width); rng=np.random.default_rng(SETTINGS['seed']); fits=[]
    for i in range(SETTINGS['starts']):
        start=problem.initial.copy()
        if i: start=np.clip(start+rng.normal(0,.025,len(start))*(problem.upper-problem.lower),problem.lower+1e-5,problem.upper-1e-5)
        fits.append(problem.fit(start))
    best=min(fits,key=lambda f:f.cost); perturb=[]
    for i in range(SETTINGS['perturbations']):
        noise=np.random.default_rng(1706+i)
        changed={v:{k:(np.array(xy)+noise.uniform(-1,1,2)).tolist() for k,xy in pts.items()} for v,pts in observations.items()}
        perturb.append(Problem(changed,sizes,normals,width).fit(best.x))
    width_runs=[]
    for factor in [.99,1.01]:
        changed_problem=Problem(observations,sizes,normals,width*factor)
        start=best.x.copy(); start[:len(problem.names)]*=factor
        for index in range(len(problem.ids)): start[len(problem.names)+index*7+3:len(problem.names)+index*7+6]*=factor
        result=changed_problem.fit(start)
        width_runs.append(dict(widthMeters=width*factor,cost=float(result.cost),success=bool(result.success),parameters=result.x.tolist()))
    p,cams=problem.unpack(best.x); used=sorted(set(k for pts in observations.values() for k in pts))
    points={k:point(k,p).tolist() for k in used}; faces=[]
    for ident,ids in FACES.items():
        if ident=='room.back': continue
        if not all(k in points for k in ids): continue
        vertices=np.array([points[k] for k in ids]); normal=np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]); length=np.linalg.norm(normal)
        if length<1e-9: continue
        normal/=length
        faces.append(dict(id=ident,entity=ident.split('.')[0],vertexIds=ids,vertices=vertices.tolist(),normal=normal.tolist(),planeOffset=float(-normal@vertices.mean(axis=0)),state='inferred',role='visual-only'))
    predictions={}; pixel_errors=[]; positive=True; normal_errors=[]
    for ident in problem.ids:
        names=list(observations[ident]); k,pose=cams[ident]; uv,z=project([points[n] for n in names],k,pose)
        errors=np.linalg.norm(uv-np.array(list(observations[ident].values())),axis=1); pixel_errors.extend(errors); positive=positive and bool(np.all(z>.05))
        predictions[ident]=dict(points=dict(zip(names,uv.tolist())),rmsePixels=float(np.sqrt(np.mean(errors**2))),p95Pixels=float(np.percentile(errors,95)))
        for normal in normals.get(ident,[]): normal_errors.append(math.degrees(math.acos(float(np.clip(np.max(np.abs(pose[:,:3].T@normal)),0,1)))))
    jac=finite_jacobian(problem,best.x); rank=rank_report(jac)
    _,_,vh=np.linalg.svd(jac,full_matrices=True)
    null_loading=np.sum(vh[rank['rank']:,:]**2,axis=0) if rank['nullity'] else np.zeros(len(best.x))
    for face in faces:
        face['sourcePoints']=[dict(view=v,pointIds=[k for k in face['vertexIds'] if k in observations[v]]) for v in problem.ids if any(k in observations[v] for k in face['vertexIds'])]
        face['uncertainty']=dict(status='underconstrained' if rank['nullity'] else 'conditional',globalJacobianNullity=rank['nullity'],authoritative=False)
    contact=[problem.labels[i] for i,x in enumerate(best.x) if min(x-problem.lower[i],problem.upper[i]-x)/(problem.upper[i]-problem.lower[i])<SETTINGS['boundFraction']]
    rmse=float(np.sqrt(np.mean(np.asarray(pixel_errors)**2))); p95=float(np.percentile(pixel_errors,95))
    provisional=rank['nullity']==0 and not contact and bool(best.success) and positive and rmse<=3 and p95<=6
    for face in faces: face['uncertainty']['status']='conditional' if provisional else 'underconstrained'
    # The opening face is a reference aperture, not a solid occluding wall.
    structure=dict(schemaVersion=1,status='provisional-fit' if provisional else 'underconstrained',coordinates=dict(frame='scene-y-up',units='meters',handedness='right',openingBottomMidpoint=[0,0,0],axes=dict(x='mouth-left-to-right',y='up',z='toward-room')),anchor=dict(kind='opening-mouth-width',meters=width,pointIds=ANCHORS),points=points,surfaces=faces,cameras=[dict(id=ident,width=sizes[ident][0],height=sizes[ident][1],intrinsics=cams[ident][0].tolist(),worldToCamera=cams[ident][1].tolist()) for ident in problem.ids],unknowns=['opening.recessDepth','occluded geometry','unlabeled props','installation accuracy','room.back full patch omitted to avoid filling opening aperture']+[k for k in SPECS if k not in problem.names],hypotheses=['rectangular planar Manhattan surfaces','centered principal point, square pixels, zero distortion','per-view unknown focal and camera pose','normal signs treated axially','only observed point IDs exported'],role='visual-only',authority='inferred',parameters={name:(float(best.x[i]) if null_loading[i]<1e-6 else None) for i,name in enumerate(problem.names)})
    structure['unknowns'] += [name for i,name in enumerate(problem.names) if null_loading[i]>=1e-6]
    std=None
    if rank['nullity']==0:
        variance=float(np.dot(best.fun,best.fun)/max(1,len(best.fun)-len(best.x)))
        std=np.sqrt(np.maximum(0,np.diag(np.linalg.pinv(jac.T@jac))*variance)).tolist()
    all_parameters=np.array([f.x for f in fits+perturb])
    diagnostics=dict(settings=SETTINGS,rank=rank,boundContacts=contact,positiveDepths=positive,rmsePixels=rmse,p95Pixels=p95,normalAngularMeanDegrees=float(np.mean(normal_errors)) if normal_errors else None,normalSigns='axial-only; signed consistency not established',conditionalParameterStd=std,uncertaintyNote='Local Jacobian covariance only if full rank; model and correspondence uncertainty excluded. Nullity or bound contact means unresolved geometry.',parameters=[dict(name=n,value=float(best.x[i]),identifiable=bool(null_loading[i]<1e-6),nullspaceLoading=float(null_loading[i]),bounds=[float(problem.lower[i]),float(problem.upper[i])],runRange=[float(all_parameters[:,i].min()),float(all_parameters[:,i].max())]) for i,n in enumerate(problem.labels)],runs=[dict(kind='start' if i<len(fits) else 'pixel-perturbation',cost=float(f.cost),success=bool(f.success),nfev=f.nfev,parameters=f.x.tolist()) for i,f in enumerate(fits+perturb)],numericalGatePassed=provisional,depthTerms=False,widthPerturbationRuns=width_runs)
    return structure,predictions,diagnostics

class ReadGuard:
    def __init__(self, inputs, output, code):
        self.inputs={Path(p).resolve() for p in inputs}; self.output=Path(output).resolve(); self.code=Path(code).resolve()
        self.runtime=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]; self.denied=[]
    def inside(self,p,root): return p==root or root in p.parents
    def __call__(self,event,args):
        if event in ['subprocess.Popen','os.system','socket.connect','socket.bind']:
            self.denied.append(event); raise PermissionError('Structural worker denies '+event)
        if event not in ['open','os.listdir','os.scandir']: return
        raw=args[0]
        if isinstance(raw,int): return
        p=Path(os.fsdecode(raw)).resolve(); mode=args[1] if event=='open' else None
        flags=args[2] if event=='open' and len(args)>2 else 0
        writing=(isinstance(mode,str) and any(m in mode for m in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        allowed=self.inside(p,self.output) if writing else p in self.inputs or self.inside(p,self.output) or self.inside(p,self.code) or any(self.inside(p,r) for r in self.runtime)
        if event!='open': allowed=self.inside(p,self.output) or self.inside(p,self.code) or any(self.inside(p,r) for r in self.runtime)
        if not allowed:
            self.denied.append(str(p)); raise PermissionError('Structural worker read/write outside allowlist: '+str(p))

def obj_text(structure):
    lines=['# Inferred visual-only visible structural patches; mouth is aperture reference']
    index=1
    for face in structure['surfaces']:
        if face['id']=='opening.mouth': continue
        lines.append('o '+face['id'])
        for xyz in face['vertices']: lines.append('v '+' '.join(format(x,'.10g') for x in xyz))
        lines.append('f '+' '.join(str(index+j) for j in range(4))); index+=4
    return '\n'.join(lines)+'\n'

def peak_memory_bytes():
    if os.name!='nt': return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    data=Counters(); data.cb=ctypes.sizeof(data)
    process=ctypes.windll.kernel32.GetCurrentProcess; process.restype=ctypes.c_void_p
    read=ctypes.windll.psapi.GetProcessMemoryInfo; read.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    return int(data.PeakWorkingSetSize) if read(process(),ctypes.byref(data),data.cb) else None

def main():
    started=time.perf_counter()
    parser=argparse.ArgumentParser(); parser.add_argument('--input',required=True); parser.add_argument('--annotations'); parser.add_argument('--lane',choices=['automatic','assisted','ablation'],required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); args.annotations=args.annotations or ('endpoints.json' if args.lane=='automatic' else 'annotations.json'); root=Path(args.input).resolve(); output=Path(args.output).resolve(); code=Path(__file__).resolve().parent
    if args.lane=='automatic' and args.annotations!='endpoints.json': raise ValueError('Automatic requires the physically separate endpoints.json file')
    if Path(args.annotations).name!=args.annotations: raise ValueError('Annotations must be an allowlisted basename')
    if output==root or root in output.parents or output in root.parents: raise ValueError('Input/output overlap')
    selected=[root/f'view-{i:02d}.png' for i in range(1,4)]+[root/args.annotations,root/'constraint.json']
    if args.lane!='ablation': selected += [root/'hints.npz',root/'manifest.json']
    output.mkdir(parents=True,exist_ok=True)
    if any(p.resolve().parent!=root for p in selected): raise ValueError('Input link escapes allowlisted input workspace')
    guard=ReadGuard(selected,output,code); sys.addaudithook(guard)
    denied_probes=[]
    for forbidden in [root/'view-04.png',root/'truth.json',root.parent/'fixture.py']:
        try:
            with open(forbidden,'rb'): pass
        except PermissionError: denied_probes.append(forbidden.name)
        else: raise RuntimeError('Audit guard failed')
    files={p.name:digest(p) for p in selected}
    code_hashes={'fit.py':digest(code/'fit.py')}
    annotation=json.loads((root/args.annotations).read_text(encoding='utf-8-sig'))
    constraint=json.loads((root/'constraint.json').read_text(encoding='utf-8-sig'))
    if set(constraint)!={'kind','width_m'} or constraint['kind']!='opening-mouth-width' or not isinstance(constraint['width_m'],(float,int)) or isinstance(constraint['width_m'],bool) or not math.isfinite(constraint['width_m']) or not .05<constraint['width_m']<20: raise ValueError('Invalid sole metric constraint')
    width=float(constraint['width_m'])
    images={f'view{i:02d}':np.asarray(Image.open(root/f'view-{i:02d}.png').convert('RGB')) for i in range(1,4)}
    sizes={k:(v.shape[1],v.shape[0]) for k,v in images.items()}
    observations=validate_annotations(annotation,sizes,automatic=args.lane=='automatic'); selection=[]
    if args.lane=='automatic': selection=automatic_points(images,observations)
    normals={}; neural=None; normal_support={}
    if args.lane!='ablation':
        neural=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
        if neural.get('neural',{}).get('independentMonocularViews') is not True or neural['neural'].get('conditioning')!={'mode':'none'} or neural['neural'].get('selectedIndices')!=[0,1,2]: raise ValueError('Neural provenance must be independent unconditioned first three views')
        if neural['neural'].get('file')!='hints.npz' or neural['neural'].get('sha256')!=files['hints.npz']: raise ValueError('Neural bytes do not match provenance')
        if len(neural.get('images',[]))!=3: raise ValueError('Expected three image provenance records')
        for i,record in enumerate(neural['images']):
            name=f'view-{i+1:02d}.png'
            if record.get('file')!=name or record.get('sha256')!=files[name] or record.get('sourceIndex')!=i: raise ValueError('Image bytes do not match neural provenance')
        with np.load(root/'hints.npz',allow_pickle=False) as hints:
            ns=hints['normals']; masks=hints['masks']
            if ns.ndim!=4 or ns.shape[0]!=3 or ns.shape[-1]!=3 or masks.shape!=ns.shape[:-1]: raise ValueError('Expected three-view normal/mask arrays')
            for i,ident in enumerate(images):
                normals[ident]=normal_clusters(ns[i],masks[i])
                sample=ns[i,::8,::8].reshape(-1,3); sample_mask=masks[i,::8,::8].reshape(-1).astype(bool)
                accepted=sample_mask & np.isfinite(sample).all(axis=1) & (np.linalg.norm(sample,axis=1)>.5)
                normal_support[ident]=dict(sampled=len(sample),accepted=int(accepted.sum()),excluded=int((~accepted).sum()),clusters=normals[ident].tolist(),axialSignAmbiguity=True)
    effective=dict(version=VERSION,lane=args.lane,settings=SETTINGS,inputHashes={k:v for k,v in files.items() if k!=args.annotations},effectiveAnnotationHash=hashlib.sha256(canonical(annotation)).hexdigest(),constraint=constraint,codeHashes=code_hashes,annotations=annotation,usedPoints=observations,neuralIdentity=neural,runtime=dict(python=platform.python_version(),pillow=Image.__version__,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__),hypothesisBounds=SPECS)
    key=hashlib.sha256(canonical(effective)).hexdigest(); receipt_path=output/'receipt.json'
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if old.get('cacheKey')!=key or hashlib.sha256(canonical(old.get('effective'))).hexdigest()!=key: raise ValueError('Existing output has a different effective identity')
        if set(old['artifacts'])!={'structure.json','predictions.json','diagnostics.json','geometry.obj'}: raise ValueError('Incomplete replay artifacts')
        for name,sha in old['artifacts'].items():
            if Path(name).name!=name or digest(output/name)!=sha: raise ValueError('Replay artifact corruption')
        print(json.dumps(dict(cacheKey=key,replayed=True,cached=True,status=old['status']))); return
    if any(output.iterdir()): raise ValueError('Unmanifested output directory is nonempty')
    structure,predictions,diagnostics=solve(observations,sizes,normals,width); diagnostics['automaticSelection']=selection; diagnostics['normalSupport']=normal_support; diagnostics['deniedReadProbes']=denied_probes
    if files!={p.name:digest(p) for p in selected} or code_hashes!={'fit.py':digest(code/'fit.py')}: raise RuntimeError('Effective input/code changed during fitting')
    diagnostics['runtime']=dict(elapsedSeconds=time.perf_counter()-started,peakWorkingSetBytes=peak_memory_bytes(),device='cpu')
    artifacts={}
    for name,value in [('structure.json',structure),('predictions.json',predictions),('diagnostics.json',diagnostics)]:
        (output/name).write_bytes(canonical(value)); artifacts[name]=digest(output/name)
    (output/'geometry.obj').write_text(obj_text(structure),encoding='utf8'); artifacts['geometry.obj']=digest(output/'geometry.obj')
    receipt=dict(schemaVersion=1,cacheKey=key,effective=effective,artifacts=artifacts,status=structure['status'],role='visual-only',paidExecution=False)
    receipt_path.write_bytes(canonical(receipt)); print(json.dumps(dict(cacheKey=key,replayed=False,cached=False,status=structure['status'],rank=diagnostics['rank']['rank'],parameters=diagnostics['rank']['columns'])))

if __name__=='__main__': main()
