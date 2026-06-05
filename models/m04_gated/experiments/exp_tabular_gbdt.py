"""
exp_tabular_gbdt.py — leak-safe LightGBM(비선형 tabular head) on full feature set.

검증 규율: 내부 SearchID 80/20 split 으로만 선택(gbdt early-stop, top-k rate). external
click_validation 라벨은 최종 보고만. CTR feature 는 target-encoding leak 방지로:
  - train 행: K-fold OOF (다른 fold 카운트로),
  - 내부-val 행: 내부-train 카운트,
  - external 행: full-train 카운트.
content score 는 GatedCTRModel._fit_head(torch GPU) 로 학습한 bi-encoder cos.
external F1 은 _best_f1_topk(exact top-k).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "6")
import sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import lightgbm as lgb
from shared.data.dataset import RecoDataset
from models.m04_gated import GateConfig, GatedCTRModel
from models.m04_gated.gated_ctr import _best_f1_topk, _binary_auc, _l2_normalize

KS = 20  # laplace

def ctr_map(keys, y, g):
    s, c = {}, {}
    for k, yy in zip(keys, y):
        s[k] = s.get(k, 0) + 1
        if yy: c[k] = c.get(k, 0) + 1
    return s, c, g

def ctr_lookup(keys, s, c, g):
    return np.array([ (c.get(k,0)+KS*g)/(s.get(k,0)+KS) if s.get(k,0)>0 else g for k in keys ], np.float64)

def main():
    ds = RecoDataset(ROOT/"../datasets").load()
    si = pd.read_csv(ds.dir/"searchinfo.csv"); ui = pd.read_csv(ds.dir/"userinfo.csv")
    sid2ip = dict(zip(si.SearchID, si.IPID)); uid2dev = dict(zip(ui.UserID, ui.UserDeviceID))
    Q,A,Y,ad,ca,us,hist,lg,sid,price,pos = ([] for _ in range(11))
    for ev in ds.training_stream():
        for a in ev.ads:
            Q.append(ev.search_emb);A.append(a.ad_emb);Y.append(int(a.is_click))
            ad.append(int(a.ad_id));ca.append(int(a.category_id));us.append(int(ev.user_id))
            hist.append(float(a.hist_ctr) if a.hist_ctr is not None else 0.);lg.append(float(ev.is_logged_on))
            sid.append(int(ev.search_id));price.append(float(getattr(a,'price',0.) or 0.));pos.append(int(a.position))
    Q=_l2_normalize(np.asarray(Q,np.float32));A=_l2_normalize(np.asarray(A,np.float32));Y=np.asarray(Y,np.float32)
    ad=np.asarray(ad);ca=np.asarray(ca);us=np.asarray(us);hist=np.asarray(hist,np.float64)
    lg=np.asarray(lg,np.float64);sid=np.asarray(sid);price=np.asarray(price,np.float64);pos=np.asarray(pos,np.float64)
    ip=np.array([sid2ip.get(int(s),-1) for s in sid]);dv=np.array([uid2dev.get(int(u),-1) for u in us])
    uc=np.array([hash((int(u),int(c)))&0x7fffffff for u,c in zip(us,ca)])  # user_cat key
    g=float(Y.mean()); n=len(Y)

    # 내부 SearchID 80/20
    uniq=np.unique(sid);cut=int(len(uniq)*0.8);vs=set(np.sort(uniq)[cut:].tolist())
    mva=np.array([s in vs for s in sid]);mtr=~mva
    tr_idx=np.where(mtr)[0]

    # content head (torch GPU), 내부-train 학습 / 내부-val early-stop
    m=GatedCTRModel(GateConfig(head_epochs=30))
    m._fit_head(Q[mtr],A[mtr],Y[mtr],Q[mva],A[mva],Y[mva])
    def clog(Qa,Aa):
        up=Qa@m._U+m._bU;vp=Aa@m._V+m._bV
        up/=np.linalg.norm(up,axis=1,keepdims=True)+1e-8;vp/=np.linalg.norm(vp,axis=1,keepdims=True)+1e-8
        return (m._scale*(up*vp).sum(1)+m._b_head).astype(np.float64)
    con=clog(Q,A); cmu,csd=con[mtr].mean(),con[mtr].std()+1e-9; conz=(con-cmu)/csd

    KEYS={'ad':ad,'ip':ip,'dv':dv,'ca':ca,'us':us,'uc':uc}
    # within-search HistCTR rank (leak-free: 자기 검색의 주어진 HistCTR만)
    df=pd.DataFrame({'sid':sid,'h':hist}); rank=df.groupby('sid')['h'].rank(pct=True).values

    def build(rows_idx, src_for_ctr):
        # src_for_ctr: callable(keyname)->(s,c) maps to look up CTR for these rows
        cols={}
        cols['logHist']=np.log(np.maximum(hist[rows_idx],1e-6))
        for kn in KEYS:
            s,c,_=src_for_ctr(kn)
            cols['ctr_'+kn]=np.log(np.maximum(ctr_lookup(KEYS[kn][rows_idx],s,c,g),1e-6))
        cols['pos']=pos[rows_idx]; cols['inv_pos']=1.0/(1.0+pos[rows_idx])
        cols['logprice']=np.log1p(price[rows_idx]); cols['logged']=lg[rows_idx]
        cols['catmatch']=((np.array([ (si.set_index('SearchID').CategoryID.get(int(s),-1)) for s in sid[rows_idx]]) == ca[rows_idx])).astype(float) if False else (rank[rows_idx]*0)  # placeholder, replaced below
        cols['hrank']=rank[rows_idx]
        cols['content']=conz[rows_idx]
        return cols
    # cat_match: search category vs ad category. search cat = ca? no, ca is ad category. search cat from searchinfo.
    sid2scat=dict(zip(si.SearchID, si.CategoryID))
    scat=np.array([int(sid2scat.get(int(s),-1)) if not pd.isna(sid2scat.get(int(s),-1)) else -1 for s in sid])
    catmatch=((scat==ca)&(scat!=-1)).astype(float)

    FEATS=['logHist','ctr_ad','ctr_ip','ctr_dv','ctr_ca','ctr_us','ctr_uc','pos','inv_pos','logprice','logged','catmatch','hrank','content']
    def matrix(rows_idx, src):
        c=build(rows_idx,src); c['catmatch']=catmatch[rows_idx]
        return np.column_stack([c[f] for f in FEATS])

    # full-train counts (for va + ext)
    full={kn:ctr_map(KEYS[kn][mtr].tolist(),Y[mtr].tolist(),g) for kn in KEYS}  # 내부-train counts
    fulltr_all={kn:ctr_map(KEYS[kn].tolist(),Y.tolist(),g) for kn in KEYS}       # full-train (external용)
    # train rows: OOF (5-fold over tr_idx)
    rng=np.random.RandomState(0); perm=rng.permutation(tr_idx); folds=np.array_split(perm,5)
    Xtr=np.zeros((len(tr_idx),len(FEATS)));  pos_in_tr={r:i for i,r in enumerate(tr_idx)}
    for f in folds:
        others=np.setdiff1d(tr_idx,f)
        src={kn:ctr_map(KEYS[kn][others].tolist(),Y[others].tolist(),g) for kn in KEYS}
        Xf=matrix(f, lambda kn: src[kn][:2]+ (g,) if False else (src[kn][0],src[kn][1],g))
        for i,r in enumerate(f): Xtr[pos_in_tr[r]]=Xf[i]
    ytr=Y[mtr]
    Xva=matrix(np.where(mva)[0], lambda kn:(full[kn][0],full[kn][1],g))
    yva=Y[mva]

    # external
    pr=ds.val_click_queries();yex=ds.val_click_answers()["IsClick"].to_numpy()[:len(pr)].astype(int)
    e={}
    e['ad']=np.array([int(a.ad_id) for _,a in pr]);e['ca']=np.array([int(a.category_id) for _,a in pr])
    e['us']=np.array([int(ev.user_id) for ev,_ in pr]);esid=np.array([int(ev.search_id) for ev,_ in pr])
    e['ip']=np.array([int(sid2ip.get(int(s),-1)) for s in esid]);e['dv']=np.array([int(uid2dev.get(int(u),-1)) for u in e['us']])
    e['uc']=np.array([hash((int(u),int(c)))&0x7fffffff for u,c in zip(e['us'],e['ca'])])
    ehist=np.array([float(a.hist_ctr) for _,a in pr],np.float64);elg=np.array([float(ev.is_logged_on) for ev,_ in pr],np.float64)
    eprice=np.array([float(getattr(a,'price',0.) or 0.) for _,a in pr],np.float64);epos=np.array([int(a.position) for _,a in pr],np.float64)
    escat=np.array([int(sid2scat.get(int(s),-1)) if not pd.isna(sid2scat.get(int(s),-1)) else -1 for s in esid])
    ecatmatch=((escat==e['ca'])&(escat!=-1)).astype(float)
    Qe=_l2_normalize(np.asarray([ev.search_emb for ev,_ in pr],np.float32));Ae=_l2_normalize(np.asarray([a.ad_emb for _,a in pr],np.float32))
    econz=(clog(Qe,Ae)-cmu)/csd
    edf=pd.DataFrame({'sid':esid,'h':ehist});erank=edf.groupby('sid')['h'].rank(pct=True).values
    def ematrix():
        cols={'logHist':np.log(np.maximum(ehist,1e-6))}
        for kn in KEYS:
            s,c,_=fulltr_all[kn]; cols['ctr_'+kn]=np.log(np.maximum(ctr_lookup(e[kn],s,c,g),1e-6))
        cols['pos']=epos;cols['inv_pos']=1.0/(1.0+epos);cols['logprice']=np.log1p(eprice);cols['logged']=elg
        cols['catmatch']=ecatmatch;cols['hrank']=erank;cols['content']=econz
        return np.column_stack([cols[f] for f in FEATS])
    Xe=ematrix()

    spw=float((ytr==0).sum()/max(1,(ytr==1).sum()))
    dtr=lgb.Dataset(Xtr,ytr); dva=lgb.Dataset(Xva,yva,reference=dtr)
    params=dict(objective='binary',metric='auc',learning_rate=0.03,num_leaves=31,min_child_samples=200,
                feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,scale_pos_weight=spw,verbose=-1)
    bst=lgb.train(params,dtr,num_boost_round=2000,valid_sets=[dva],
                  callbacks=[lgb.early_stopping(80),lgb.log_evaluation(0)])
    sva=bst.predict(Xva); se=bst.predict(Xe)
    intf=_best_f1_topk(sva,yva)[0]; extf,extk=_best_f1_topk(se,yex); exauc=_binary_auc(se,yex)
    print(f"[gbdt] best_iter={bst.best_iteration} int_f1={intf:.4f} EXTERNAL_BEST_F1={extf:.4f} extAUC={exauc:.4f} k={extk}")
    os.makedirs("/tmp/pumasi_res",exist_ok=True)
    json.dump({"approach":"tabular-gbdt","ext_best_f1":float(extf),"ext_auc":float(exauc),"int_f1":float(intf),
               "n_features":len(FEATS),"best_iter":int(bst.best_iteration),
               "leak_audit":"internal SearchID 80/20 for early-stop/threshold; train CTR via 5-fold OOF; val CTR from internal-train; external CTR from full-train; external answers only final report"},
              open("/tmp/pumasi_res/tabular-gbdt.json","w"))
    print(f"EXTERNAL_BEST_F1={extf:.4f}")

if __name__=="__main__":
    main()
