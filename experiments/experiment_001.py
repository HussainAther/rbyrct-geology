from pathlib import Path
import sys,time,json
import numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); from src.core import *
def main():
    img,inc,center,rock=make_phantom(); a=angles(); sino=acquire(img,a); rng=np.random.default_rng(2026); rows=[]; cache={}; budgets=[5,10,20,40,100]
    for pct in budgets:
        n=max(6,round(len(a)*pct/100)); n=min(n,len(a))
        for name,fn in [('uniform_sparse',lambda:uniform_idx(len(a),n)),('random_sparse',lambda:random_idx(len(a),n,rng)),('adaptive',lambda:adaptive_idx(sino,a,n,img.shape[0]))]:
            t=time.perf_counter(); idx=fn(); elapsed=time.perf_counter()-t; rec=reconstruct(sino[:,idx],a[idx],img.shape[0]); m,s,cnr,iou,loc=metrics(img,rec,inc,center,rock); rows.append(dict(strategy=name,budget_pct=pct,angles=len(idx),ray_count=sino.shape[0]*len(idx),mse=m,ssim=s,cnr=cnr,inclusion_iou=iou,localization_error_px=loc,runtime_s=elapsed)); cache[(name,pct)]=rec
    rec=reconstruct(sino,a,img.shape[0]); m,s,cnr,iou,loc=metrics(img,rec,inc,center,rock); rows.append(dict(strategy='dense_reference',budget_pct=100,angles=len(a),ray_count=sino.shape[0]*len(a),mse=m,ssim=s,cnr=cnr,inclusion_iou=iou,localization_error_px=loc,runtime_s=0))
    df=pd.DataFrame(rows); df.to_csv(ROOT/'results/metrics.csv',index=False); (ROOT/'results/summary.json').write_text(json.dumps({'budgets_pct':budgets},indent=2))
    fig,axs=plt.subplots(1,4,figsize=(12,3)); panels=[('Ground truth',img),('Uniform 20%',cache[('uniform_sparse',20)]),('Random 20%',cache[('random_sparse',20)]),('Adaptive 20%',cache[('adaptive',20)])]
    for ax,(t,x) in zip(axs,panels): ax.imshow(x,cmap='gray',vmin=0,vmax=1); ax.set_title(t); ax.axis('off')
    fig.tight_layout(); fig.savefig(ROOT/'figures/reconstruction_comparison.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    for st in ['uniform_sparse','random_sparse','adaptive']:
        d=df[df.strategy==st].sort_values('ray_count'); ax.plot(d.ray_count,d.cnr,marker='o',label=st)
    ax.set(xlabel='Acquired rays',ylabel='Inclusion CNR',title='Geological inclusion detectability'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/cnr_vs_ray_budget.png',dpi=180); plt.close(fig)
    print(df.to_string(index=False))
if __name__=='__main__':main()
