import numpy as np, pandas as pd, sys
from pathlib import Path
ROOT=Path("/home/gyuch_an02/marijuana/Data-Mining-term-project");sys.path.insert(0,str(ROOT))
from models.m04_gated.gated_ctr import _best_f1_topk,_fit_f1_exponents,_binary_auc
d=np.load("/tmp/cs_full.npz")
con,cone,Y=d['con'],d['cone'],d['Y'];ad,ip,dvk,ca,uid,sid,hist,lg=d['ad'],d['ip'],d['dvk'],d['ca'],d['uid'],d['sid'],d['hist'],d['lg']
mtr,mva,yex=d['mtr'],d['mva'],d['yex'];euid,esid,eca,ead,ehist,elg=d['euid'],d['esid'],d['eca'],d['ead'],d['ehist'],d['elg']
si=pd.read_csv(ROOT/"../datasets/searchinfo.csv");ui=pd.read_csv(ROOT/"../datasets/userinfo.csv")
s2ip=dict(zip(si.SearchID,si.IPID));u2d=dict(zip(ui.UserID,ui.UserDeviceID))
eip=np.array([s2ip.get(int(s),-1) for s in esid]);edv=np.array([u2d.get(int(u),-1) for u in euid])
g=float(Y.mean());KS=20;LO=np.log(1.7);prev=g
def cs(keys,y):
    df=pd.DataFrame({'k':keys,'y':y});gp=df.groupby('k')['y'].agg(['sum','count']);return (gp['sum']+KS*g)/(gp['count']+KS)
def look(keys,ser): return np.log(np.maximum(pd.Series(keys).map(ser).fillna(g).values,1e-6))
KE={'ad':(ad,ead),'ip':(ip,eip),'dv':(dvk,edv),'ca':(ca,eca)}
cmt={k:cs(KE[k][0][mtr],Y[mtr]) for k in KE};cmf={k:cs(KE[k][0],Y) for k in KE}
cmu,csd=con[mtr].mean(),con[mtr].std()+1e-9
vi=np.where(mva)[0]
Xv=np.column_stack([np.log(np.maximum(hist[vi],1e-6))]+[look(KE[k][0][vi],cmt[k]) for k in KE]+[(con[vi]-cmu)/csd])
bv=LO*(1-lg[vi]);w=_fit_f1_exponents(Xv,Y[vi],base=bv)
Xe=np.column_stack([np.log(np.maximum(ehist,1e-6))]+[look(KE[k][1],cmf[k]) for k in KE]+[(cone-cmu)/csd])
se=Xe@w+LO*(1-elg)
# 층화: 내부-train(mtr) 유저 이력
ushow=pd.Series(uid[mtr]).value_counts().to_dict()
uclk=pd.Series(uid[mtr][Y[mtr]==1]).value_counts().to_dict()
strat=np.array(['cold' if u not in ushow else ('warm' if uclk.get(u,0)>0 else 'search-only') for u in euid])
# 전역 prevalence top-k 예측
k=max(1,int(round(prev*len(se))));order=np.argsort(-se);pred=np.zeros(len(se),int);pred[order[:k]]=1
print(f"전역: k={k} (prev {prev:.4f}) external F1={ _best_f1_topk(se,yex)[0]:.4f}(oracle) / honest@prev 아래")
def f1p(p,y):
    tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
    pr=tp/(tp+fp) if tp+fp else 0.;rc=tp/(tp+fn) if tp+fn else 0.;f=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
    return tp,fp,fn,pr,rc,f
out=open("/tmp/strat_m04.txt","w")
out.write("stratum     n      share  pos  pos%   AUC    TP FP FN  prec    rec     F1\n")
TPg=int(((pred==1)&(yex==1)).sum())
for s in ['cold','search-only','warm']:
    m=strat==s;n=int(m.sum());p=int(yex[m].sum())
    tp,fp,fn,pr,rc,f=f1p(pred[m],yex[m])
    try: au=_binary_auc(se[m],yex[m])
    except: au=float('nan')
    out.write(f"{s:11s} {n:6d} {n/len(yex):6.1%} {p:4d} {p/max(1,n):5.2%} {au:.3f}  {tp:2d} {fp:3d} {fn:3d} {pr:6.2%} {rc:6.2%} {f:.4f}\n")
gtp,gfp,gfn,gpr,grc,gf=f1p(pred,yex)
out.write(f"{'전체':11s} {len(yex):6d} {1.0:6.1%} {int(yex.sum()):4d} {yex.mean():5.2%} {_binary_auc(se,yex):.3f}  {gtp:2d} {gfp:3d} {gfn:3d} {gpr:6.2%} {grc:6.2%} {gf:.4f}\n")
out.close()
print(open("/tmp/strat_m04.txt").read())
