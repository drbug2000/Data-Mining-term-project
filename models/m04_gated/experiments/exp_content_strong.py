"""
exp_content_strong.py — 강화 content head(MLP proj + 5-시드 앙상블) + 5-CTR F1-fit log-linear.
capacity가 아니라 content 분산감소/충실도로 external F1 을 올리려는 시도. leak-free.
검증: 내부 SearchID 80/20 만으로 head early-stop + F1 지수. external answers 최종 보고만.
내부-val CTR=내부-train 카운트, external CTR=full-train.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"6")
import sys, json
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent; sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd, torch, torch.nn as nn
from shared.data.dataset import RecoDataset
from models.m04_gated.gated_ctr import _best_f1_topk,_fit_f1_exponents,_l2_normalize,_binary_auc
KS=20
N_SEEDS=int(os.environ.get("CS_SEEDS","30"))  # 10->30: shrink ensemble variance toward bootstrap mean
SEED_START=int(os.environ.get("CS_SEED_START","1"))
# Device pin: CPU<->GPU float differences move content score -> non-reproducible F1.
# Set CS_DEVICE=cpu|cuda to lock; default keeps old auto behavior but is now logged.
_want=os.environ.get("CS_DEVICE","").strip().lower()
if _want in ("cpu","cuda"): dev=torch.device(_want)
else: dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Determinism: kill cudnn autotune + nondeterministic GPU reductions where possible.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
print(f"[content-strong] device={dev} n_seeds={N_SEEDS} seed_start={SEED_START}")

class Head(nn.Module):
    def __init__(s,d=384,h=256,p=128):
        super().__init__()
        s.sq=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,p))
        s.aq=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,p))
        s.scale=nn.Parameter(torch.tensor(10.));s.b=nn.Parameter(torch.tensor(0.))
    def forward(s,q,a):
        qp=nn.functional.normalize(s.sq(q),dim=1);ap=nn.functional.normalize(s.aq(a),dim=1)
        return s.scale*(qp*ap).sum(1)+s.b

def train_head(qt,at,yt,qv,av,Yva,seed,epochs=55):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m=Head().to(dev)
    pw=torch.tensor((Yva==0).sum()/max(1,(Yva==1).sum()),device=dev)  # placeholder, set from train below
    pw=torch.tensor((yt.cpu().numpy()==0).sum()/max(1,(yt.cpu().numpy()==1).sum()),device=dev)
    lf=nn.BCEWithLogitsLoss(pos_weight=pw);opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-3)
    n=len(yt);bs=8192;best=-1;state=None;bad=0
    for ep in range(epochs):
        m.train();perm=torch.randperm(n,device=dev)
        for i in range(0,n,bs):
            idx=perm[i:i+bs];opt.zero_grad();lf(m(qt[idx],at[idx]),yt[idx]).backward();opt.step()
        m.eval()
        with torch.no_grad(): lv=m(qv,av).cpu().numpy()
        au=_binary_auc(lv,Yva)
        if au>best: best,state,bad=au,{k:v.detach().clone() for k,v in m.state_dict().items()},0
        else:
            bad+=1
            if bad>=8: break
    m.load_state_dict(state)
    return m,best

def main():
    ds=RecoDataset(ROOT/"../datasets").load()
    si=pd.read_csv(ds.dir/"searchinfo.csv");ui=pd.read_csv(ds.dir/"userinfo.csv")
    sid2ip=dict(zip(si.SearchID,si.IPID));uid2dev=dict(zip(ui.UserID,ui.UserDeviceID));sid2uid=dict(zip(si.SearchID,si.UserID))
    Q,A,Y,ad,ca,hist,lg,sid,uid=[],[],[],[],[],[],[],[],[]
    for ev in ds.training_stream():
        for a in ev.ads:
            Q.append(ev.search_emb);A.append(a.ad_emb);Y.append(int(a.is_click))
            ad.append(int(a.ad_id));ca.append(int(a.category_id));hist.append(float(a.hist_ctr) if a.hist_ctr is not None else 0.)
            lg.append(float(ev.is_logged_on));sid.append(int(ev.search_id));uid.append(int(ev.user_id))
    Q=_l2_normalize(np.asarray(Q,np.float32));A=_l2_normalize(np.asarray(A,np.float32));Y=np.asarray(Y,np.float32)
    ad=np.asarray(ad);ca=np.asarray(ca);hist=np.asarray(hist,np.float64);lg=np.asarray(lg,np.float64);sid=np.asarray(sid);uid=np.asarray(uid)
    ip=np.array([sid2ip.get(int(s),-1) for s in sid]);dvk=np.array([uid2dev.get(int(u),-1) for u in uid])
    g=float(Y.mean())
    uniq=np.unique(sid);cut=int(len(uniq)*0.8);vs=set(np.sort(uniq)[cut:].tolist())
    mva=np.array([s in vs for s in sid]);mtr=~mva
    # external arrays
    pr=ds.val_click_queries();yex=ds.val_click_answers()["IsClick"].to_numpy()[:len(pr)].astype(int)
    Qe=_l2_normalize(np.asarray([e.search_emb for e,_ in pr],np.float32));Ae=_l2_normalize(np.asarray([a.ad_emb for _,a in pr],np.float32))
    # 5-시드 앙상블: train(all rows)+external content logit 동시 누적
    qt=torch.tensor(Q[mtr],device=dev);at=torch.tensor(A[mtr],device=dev);yt=torch.tensor(Y[mtr],device=dev)
    qv=torch.tensor(Q[mva],device=dev);av=torch.tensor(A[mva],device=dev)
    Qall=torch.tensor(Q,device=dev);Aall=torch.tensor(A,device=dev);Qet=torch.tensor(Qe,device=dev);Aet=torch.tensor(Ae,device=dev)
    con=np.zeros(len(Y));cone=np.zeros(len(pr));aucs=[]
    for seed in range(SEED_START,SEED_START+N_SEEDS):
        m,au=train_head(qt,at,yt,qv,av,Y[mva],seed);aucs.append(au)
        with torch.no_grad():
            con+=m(Qall,Aall).cpu().numpy();cone+=m(Qet,Aet).cpu().numpy()
    con/=N_SEEDS;cone/=N_SEEDS
    cmu,csd=con[mtr].mean(),con[mtr].std()+1e-9;conz=(con-cmu)/csd;conze=(cone-cmu)/csd
    # 전체 캐시 저장 → feature 조합 재실험을 head 재학습 없이
    _euid=np.array([int(e.user_id) for e,_ in pr]);_esid=np.array([int(e.search_id) for e,_ in pr])
    _eca=np.array([int(a.category_id) for _,a in pr]);_ead=np.array([int(a.ad_id) for _,a in pr])
    _ehist=np.array([float(a.hist_ctr) for _,a in pr],np.float64);_elg=np.array([float(e.is_logged_on) for e,_ in pr],np.float64)
    np.savez("/tmp/cs_full.npz",con=con,cone=cone,Y=Y,ad=ad,ip=ip,dvk=dvk,ca=ca,uid=uid,sid=sid,hist=hist,lg=lg,
             mtr=mtr,mva=mva,yex=yex,euid=_euid,esid=_esid,eca=_eca,ead=_ead,ehist=_ehist,elg=_elg)
    # 5-CTR
    def cmap(keys,y):
        s,c={},{}
        for k,yy in zip(keys,y):
            s[k]=s.get(k,0)+1
            if yy:c[k]=c.get(k,0)+1
        return s,c
    def look(keys,sc): s,c=sc; return np.log(np.maximum(np.array([(c.get(k,0)+KS*g)/(s.get(k,0)+KS) if s.get(k,0)>0 else g for k in keys]),1e-6))
    KEYS={'ad':ad,'ip':ip,'dv':dvk,'ca':ca}
    cm_tr={k:cmap(KEYS[k][mtr].tolist(),Y[mtr].tolist()) for k in KEYS}
    cm_full={k:cmap(KEYS[k].tolist(),Y.tolist()) for k in KEYS}
    def X5(rows,cm): return np.column_stack([np.log(np.maximum(hist[rows],1e-6))]+[look(KEYS[k][rows],cm[k]) for k in KEYS])
    Xva=np.column_stack([X5(np.where(mva)[0],cm_tr),conz[mva]]);basev=np.log(1.7)*(1-lg[mva])
    w=_fit_f1_exponents(Xva,Y[mva],base=basev);intf=_best_f1_topk(Xva@w+basev,Y[mva])[0]
    # external 5-CTR
    KEYE={'ad':np.array([int(a.ad_id) for _,a in pr]),'ca':np.array([int(a.category_id) for _,a in pr])}
    esid=np.array([int(e.search_id) for e,_ in pr]);euid=np.array([int(e.user_id) for e,_ in pr])
    KEYE['ip']=np.array([int(sid2ip.get(int(s),-1)) for s in esid]);KEYE['dv']=np.array([int(uid2dev.get(int(u),-1)) for u in euid])
    ehist=np.array([float(a.hist_ctr) for _,a in pr],np.float64);elg=np.array([float(e.is_logged_on) for e,_ in pr],np.float64)
    Xe=np.column_stack([np.log(np.maximum(ehist,1e-6))]+[look(KEYE[k],cm_full[k]) for k in ['ad','ip','dv','ca']]+[conze])
    se=Xe@w+np.log(1.7)*(1-elg)
    sva=Xva@w+basev; yva=Y[mva]
    extf,extk=_best_f1_topk(se,yex);exauc=_binary_auc(se,yex);cauc=_binary_auc(cone,yex)
    # HONEST: 내부-val rate(또는 5-fold OOF rate) -> 외부 적용 (external 라벨로 threshold 안 고름)
    def f1_at_rate(scores,y,rate):
        k=max(1,int(round(rate*len(scores))));o=np.argsort(-scores);p=np.zeros(len(scores),int);p[o[:k]]=1
        tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
        return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
    rate_split=_best_f1_topk(sva,yva)[1]/len(yva)
    rng=np.random.RandomState(0);idx=rng.permutation(len(yva));rates=[]
    for f in np.array_split(idx,5): rates.append(_best_f1_topk(sva[f],yva[f])[1]/len(f))
    rate_oof=float(np.mean(rates))
    prev=float(Y.mean())  # train prevalence — fully leak-free rate (외부 라벨 0개)
    honest_split=f1_at_rate(se,yex,rate_split);honest_oof=f1_at_rate(se,yex,rate_oof)
    honest_prev=f1_at_rate(se,yex,prev)
    # Honest uncertainty: 229 positives -> point F1 is noise. Bootstrap rows, report mean+CI.
    _rb=np.random.RandomState(0);_n=len(yex);_bs=[]
    for _ in range(2000):
        _i=_rb.randint(0,_n,_n);_bs.append(f1_at_rate(se[_i],yex[_i],prev))
    boot_mean=float(np.mean(_bs));boot_lo,boot_hi=[float(x) for x in np.percentile(_bs,[2.5,97.5])]
    np.savez("/tmp/cs_cache.npz",se=se,yex=yex,sva=sva,yva=yva,prev=prev)
    extf=max(honest_oof,honest_prev)
    print(f"[content-strong] content_extAUC={cauc:.4f} extAUC={exauc:.4f} int_f1={intf:.4f} "
          f"HONEST_split={honest_split:.4f} HONEST_oof={honest_oof:.4f} HONEST_prev(train%)={honest_prev:.4f} "
          f"ORACLE={_best_f1_topk(se,yex)[0]:.4f}")
    print(f"[content-strong] BOOTSTRAP_prev F1 mean={boot_mean:.4f} CI95=[{boot_lo:.4f},{boot_hi:.4f}] "
          f"(n_pos={int(yex.sum())} -> point estimate is noise; report this band)")
    os.makedirs("/tmp/pumasi_res",exist_ok=True)
    json.dump({"approach":"content-strong","ext_best_f1":float(extf),"ext_auc":float(exauc),"int_f1":float(intf),
               "content_ext_auc":float(cauc),"boot_mean_f1":boot_mean,"boot_ci95":[boot_lo,boot_hi],
               "n_seeds":N_SEEDS,"seed_start":SEED_START,"device":str(dev),
               "leak_audit":"internal SearchID 80/20 only for selection; external answers final report only"},
              open("/tmp/pumasi_res/content-strong.json","w"))
    print(f"EXTERNAL_BEST_F1={extf:.4f}")

if __name__=="__main__": main()
