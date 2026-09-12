import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk,line
from skimage.transform import radon,iradon
from skimage.metrics import structural_similarity

def make_phantom(size=160,seed=14):
    rng=np.random.default_rng(seed); img=np.zeros((size,size),np.float32); c=size//2
    rr,cc=disk((c,c),int(.40*size),shape=img.shape); mask=np.zeros_like(img,bool); mask[rr,cc]=True
    tex=gaussian_filter(rng.normal(size=img.shape),sigma=4); tex=(tex-tex.mean())/(tex.std()+1e-9)
    img[mask]=.45+.10*tex[mask]
    for y,x,rad,val in [(.38,.44,.07,.63),(.61,.57,.06,.29),(.52,.31,.045,.57)]:
        rr,cc=disk((int(y*size),int(x*size)),int(rad*size),shape=img.shape); img[rr,cc]=val
    rr,cc=line(int(.25*size),int(.35*size),int(.70*size),int(.62*size));
    for dr in [-1,0,1]: img[np.clip(rr+dr,0,size-1),cc]=.18
    inclusion=np.zeros_like(img,bool); center=(int(.43*size),int(.64*size)); rr,cc=disk(center,int(.028*size),shape=img.shape); img[rr,cc]=.92; inclusion[rr,cc]=True
    img[~mask]=0; return np.clip(img,0,1),inclusion,center,mask

def angles(n=180): return np.linspace(0,180,n,endpoint=False)
def acquire(img,a): return radon(img,theta=a,circle=True)
def reconstruct(sino,a,size): return np.clip(np.nan_to_num(iradon(sino,theta=a,circle=True,filter_name='ramp',output_size=size)),0,1)
def uniform_idx(nall,n): return np.unique(np.linspace(0,nall-1,n,dtype=int))
def random_idx(nall,n,rng): return np.sort(rng.choice(nall,n,replace=False))
def _dist(a,b):
    d=np.abs(b-a); return np.minimum(d,180-d)
def adaptive_idx(sino,a,n,size,seed=6):
    selected=np.unique(np.linspace(0,len(a)-1,min(seed,n),dtype=int)).tolist()
    if len(selected)>=n:return np.array(sorted(selected))
    sel=np.array(selected); rec=reconstruct(sino[:,sel],a[sel],size)
    # Heterogeneity proxy: local high-frequency residual from smoothed pilot reconstruction.
    heter=np.abs(rec-gaussian_filter(rec,2.5))
    cand=np.array([i for i in range(len(a)) if i not in selected]); p=radon(heter,theta=a[cand],circle=True)
    info=np.mean(np.abs(np.diff(p,axis=0)),axis=0); info=(info-info.min())/(np.ptp(info)+1e-9); priority={int(i):float(v) for i,v in zip(cand,info)}
    while len(selected)<n:
        cand=np.array([i for i in range(len(a)) if i not in selected]); sa=a[np.array(selected)]
        div=np.array([np.min(_dist(a[i],sa)) for i in cand]); div/=max(div.max(),1e-9); inf=np.array([priority.get(int(i),0) for i in cand])
        selected.append(int(cand[np.argmax(.35*inf+1.0*div)]))
    return np.array(sorted(selected))
def metrics(ref,rec,inclusion,center,rockmask):
    mse=float(np.mean((ref-rec)**2)); s=float(structural_similarity(ref,rec,data_range=1.0))
    yy,xx=np.indices(ref.shape); cy,cx=center; ring=((yy-cy)**2+(xx-cx)**2<=12**2)&(~inclusion)&rockmask
    cnr=float((rec[inclusion].mean()-rec[ring].mean())/(rec[ring].std()+1e-9))
    vals=rec[rockmask]; thr=float(np.quantile(vals,.99)); pred=(rec>=thr)&rockmask
    inter=np.logical_and(pred,inclusion).sum(); union=np.logical_or(pred,inclusion).sum(); iou=float(inter/union) if union else 1.0
    score=np.where(rockmask,rec,-1); py,px=np.unravel_index(np.argmax(score),score.shape); loc=float(np.hypot(py-cy,px-cx))
    return mse,s,cnr,iou,loc
