"""
exp_koskncp.py — KoSKNCP 중심 재현: Task1(클릭예측 F1) + Task2(광고추천 NDCG@3).

KoSKNCP 핵심 = SKNCP(의미적 k-최근접 클릭예측): "나와 비슷한 검색을 한 사람들이 실제 클릭한
광고와, 지금 이 광고가 얼마나 비슷한가". 학습 content head 대신 비모수 협업필터 신호.

Task1: 9항 로그-오즈 [logHist, ad_ctr, ip_ctr, dev_ctr, cat_ctr, IPS_pos, rank, SKNCP, nbr_ctr]
       F1-목적 좌표상승(w_min=0.5 정규화), sorted 80/20, honest(OOF/prevalence) rate.
Task2: 각 검색쿼리에 대해 전체 후보광고를 SKNCP(+ad_ctr 사전) 로 랭킹 → NDCG@3
       (shared.eval.predictor.evaluate_task_b_ndcg).
검증: 외부 정답(click/ad_validation_answer)은 최종 채점만. 모든 선택은 학습 내부에서만.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"6")
import sys, json, time
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent.parent; sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd, torch
from shared.data.dataset import RecoDataset
from shared.eval.predictor import evaluate_task_b_ndcg
from models.m04_gated.gated_ctr import _best_f1_topk,_fit_f1_exponents,_l2_normalize,_binary_auc
DATA=ROOT/"../datasets"; KS=20; LO=np.log(1.7)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def L2(x): n=np.linalg.norm(x,axis=1,keepdims=True); return x/(n+1e-8)

def skncp_rows(Qs, As_rows, AD, TQ, q2adrows, K=100, cap=300, blk=512):
    """행별 SKNCP: query → K 최근접 train-클릭쿼리 → 그 clicked ad pool → max cos(현재 ad)."""
    M=len(Qs); out=np.zeros(M,np.float32)
    Qd=torch.tensor(Qs,device=dev); TQd=torch.tensor(TQ,device=dev); ADd=torch.tensor(AD,device=dev)
    for i in range(0,M,blk):
        sims=Qd[i:i+blk]@TQd.T; top=sims.topk(min(K,TQd.shape[0]),dim=1).indices.cpu().numpy()
        for j in range(top.shape[0]):
            rows=[]
            for t in top[j]:
                rows.extend(q2adrows[t])
                if len(rows)>=cap: break
            r=i+j; a=As_rows[r]
            if rows and a>=0:
                pool=ADd[torch.tensor(rows[:cap],device=dev)]
                out[r]=float((ADd[a]@pool.T).max())
    return out

def skncp_allcands(Qquery, AD_all, TQ, q2adrows, K=100, cap=300):
    """Task2: 한 쿼리 → 전체 후보광고(AD_all) 각각에 대한 SKNCP. (Ncand,) 반환."""
    qd=torch.tensor(Qquery,device=dev); TQd=torch.tensor(TQ,device=dev)
    sims=qd@TQd.T; top=sims.topk(min(K,TQd.shape[0])).indices.cpu().numpy()
    rows=[]
    for t in top:
        rows.extend(q2adrows[t])
        if len(rows)>=cap: break
    if not rows: return np.zeros(AD_all.shape[0],np.float32)
    pool=torch.tensor(AD_all[rows[:cap]],device=dev)
    return (torch.tensor(AD_all,device=dev)@pool.T).max(dim=1).values.cpu().numpy()

def main():
    t0=time.time()
    ds=RecoDataset(DATA).load()
    si=pd.read_csv(ds.dir/"searchinfo.csv");ui=pd.read_csv(ds.dir/"userinfo.csv");adi=pd.read_csv(ds.dir/"adinfo.csv")
    s2ip=dict(zip(si.SearchID,si.IPID));u2d=dict(zip(ui.UserID,ui.UserDeviceID))
    SE=L2(np.load(ds.dir/"searchinfo_text_embs.npy").astype(np.float32))
    AD=L2(np.load(ds.dir/"adinfo_title_embs.npy").astype(np.float32))
    sid2row={int(s):i for i,s in enumerate(si.SearchID.values)}; adid2row={int(a):i for i,a in enumerate(adi.AdID.values)}
    # collect train
    ad,ca,uid,sid,hist,lg,pos,Y=[],[],[],[],[],[],[],[]
    for ev in ds.training_stream():
        for a in ev.ads:
            ad.append(int(a.ad_id));ca.append(int(a.category_id));uid.append(int(ev.user_id));sid.append(int(ev.search_id))
            hist.append(float(a.hist_ctr) if a.hist_ctr is not None else 0.);lg.append(float(ev.is_logged_on));pos.append(int(a.position));Y.append(int(a.is_click))
    ad=np.array(ad);ca=np.array(ca);uid=np.array(uid);sid=np.array(sid);hist=np.array(hist);lg=np.array(lg);pos=np.array(pos);Y=np.array(Y,np.float32)
    ip=np.array([s2ip.get(int(s),-1) for s in sid]);dvk=np.array([u2d.get(int(u),-1) for u in uid]);g=float(Y.mean());n=len(Y)
    qrow=np.array([sid2row.get(int(s),-1) for s in sid]);arow=np.array([adid2row.get(int(a),-1) for a in ad])
    uniq=np.sort(np.unique(sid));cut=int(len(uniq)*0.8);vs=set(uniq[cut:].tolist())
    mva=np.array([s in vs for s in sid]);mtr=~mva
    print(f"[koskncp] loaded n={n} ({time.time()-t0:.0f}s)")

    # SKNCP 인덱스: 클릭 쿼리(검색emb) → 클릭 광고 row
    def idx(mask):
        clk=np.where(mask & (Y==1))[0]
        return SE[qrow[clk]], [[arow[r]] for r in clk]
    TQ_itr,rows_itr=idx(mtr); TQ_full,rows_full=idx(np.ones(n,bool))

    # ---------- Task1 ----------
    pr=ds.val_click_queries(); yex=ds.val_click_answers()["IsClick"].to_numpy()[:len(pr)].astype(int)
    eQrow=np.array([sid2row.get(int(e.search_id),-1) for e,_ in pr]); eArow=np.array([adid2row.get(int(a.ad_id),-1) for _,a in pr])
    eQ=SE[eQrow]; eAD_idx=eArow
    sk_va=skncp_rows(SE[qrow[mva]],arow[mva],AD,TQ_itr,rows_itr)
    sk_ex=skncp_rows(eQ,eArow,AD,TQ_full,rows_full)
    print(f"[koskncp] Task1 SKNCP done ({time.time()-t0:.0f}s)")
    ead=np.array([int(a.ad_id) for _,a in pr]);eca=np.array([int(a.category_id) for _,a in pr])
    euid=np.array([int(e.user_id) for e,_ in pr]);esid=np.array([int(e.search_id) for e,_ in pr])
    eip=np.array([s2ip.get(int(s),-1) for s in esid]);edv=np.array([u2d.get(int(u),-1) for u in euid])
    ehist=np.array([float(a.hist_ctr) for _,a in pr]);elg=np.array([float(e.is_logged_on) for e,_ in pr]);epos=np.array([int(a.position) for _,a in pr])
    def cs(k,y): df=pd.DataFrame({'k':k,'y':y});gp=df.groupby('k')['y'].agg(['sum','count']);return (gp['sum']+KS*g)/(gp['count']+KS)
    def lk(k,s): return np.log(np.maximum(pd.Series(k).map(s).fillna(g).values,1e-6))
    KE={'ad':(ad,ead),'ip':(ip,eip),'dv':(dvk,edv),'ca':(ca,eca)}
    cmt={k:cs(KE[k][0][mtr],Y[mtr]) for k in KE};cmf={k:cs(KE[k][0],Y) for k in KE}
    cps_tr=cs(pos[mtr],Y[mtr]);cps_f=cs(pos,Y)
    def logit(x): x=np.clip(x,1e-6,1-1e-6); return np.log(x/(1-x))
    def feats(split):
        if split=='va':
            r=np.where(mva)[0];A_=ad[r];IPk=ip[r];DV=dvk[r];C=ca[r];H=hist[r];P=pos[r];S=sid[r];SK=sk_va;cm=cmt;cp=cps_tr;LGv=lg[r]
        else:
            A_=ead;IPk=eip;DV=edv;C=eca;H=ehist;P=epos;S=esid;SK=sk_ex;cm=cmf;cp=cps_f;LGv=elg
        adc=lk(A_,cm['ad']);ipc=lk(IPk,cm['ip']);dvc=lk(DV,cm['dv']);cac=lk(C,cm['ca'])
        posc=np.log(np.maximum(pd.Series(P).map(cp).fillna(g).values,1e-6));ips=posc-np.log(g)
        rk=pd.DataFrame({'s':S,'h':H}).groupby('s')['h'].rank(pct=True).values;rankf=0.5-rk
        nbr=logit(np.exp(adc))-logit(g)
        X=np.column_stack([np.log(np.maximum(H,1e-6)),adc,ipc,dvc,cac,ips,rankf,SK,nbr])
        return X,LO*(1-LGv)
    Xv,bv=feats('va');yv=Y[mva];Xe,be=feats('ex')
    w=_fit_f1_exponents(Xv,yv,base=bv,grid=np.arange(0.5,3.05,0.5))  # w_min=0.5
    sv=Xv@w+bv;se=Xe@w+be
    rng=np.random.RandomState(0); idx5=rng.permutation(len(yv))
    roof=float(np.mean([_best_f1_topk(sv[f],yv[f])[1]/len(f) for f in np.array_split(idx5,5)]))
    def f1r(s,y,r): k=max(1,int(round(r*len(s))));o=np.argsort(-s);p=np.zeros(len(s),int);p[o[:k]]=1; tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum()); return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
    t1_oof=f1r(se,yex,roof);t1_prev=f1r(se,yex,g);t1_orc=_best_f1_topk(se,yex)[0];t1_auc=_binary_auc(se,yex)
    NAMES=['logHist','ad_ctr','ip_ctr','dev_ctr','cat_ctr','IPS','rank','SKNCP','nbr']
    print(f"\n[TASK1] honest_oof={t1_oof:.4f} honest_prev={t1_prev:.4f} oracle={t1_orc:.4f} AUC={t1_auc:.4f}")
    print("  w:",{NAMES[i]:round(float(w[i]),2) for i in range(len(NAMES))})

    # ---------- Task2 (NDCG@3) ----------
    cand_emb,cand_ids=ds.all_ad_embs(); cand_emb=L2(cand_emb.astype(np.float32))
    vadq=ds.val_ad_queries(); vada=ds.val_ad_answers()
    adctr_full=cs(ad,Y)  # 전역 ad_ctr (사전)
    cand_adctr=np.array([adctr_full.get(int(a),g) for a in cand_ids])
    cand_logadctr=np.log(np.maximum(cand_adctr,1e-6))
    # 여러 점수 방식 비교
    def task2(alpha):  # score = SKNCP + alpha*log(ad_ctr)
        out={}
        for ev in vadq:
            qe=SE[sid2row[int(ev.search_id)]]
            sk=skncp_allcands(qe,cand_emb,TQ_full,rows_full)
            out[ev.search_id]=sk+alpha*cand_logadctr
        return out
    def task2_cos():  # baseline: raw query-ad cosine
        return {ev.search_id: cand_emb@SE[sid2row[int(ev.search_id)]] for ev in vadq}
    r_sk=evaluate_task_b_ndcg(task2(0.0),vada,cand_ids)
    r_skad=evaluate_task_b_ndcg(task2(0.3),vada,cand_ids)
    r_cos=evaluate_task_b_ndcg(task2_cos(),vada,cand_ids)
    print(f"\n[TASK2 NDCG@3] SKNCP-only={r_sk['ndcg@3']:.4f}  SKNCP+ad_ctr={r_skad['ndcg@3']:.4f}  raw-cosine(baseline)={r_cos['ndcg@3']:.4f}  (n={r_sk['n_queries']})")

    os.makedirs("/tmp/pumasi_res",exist_ok=True)
    json.dump({"approach":"koskncp",
               "task1_honest_oof":t1_oof,"task1_honest_prev":t1_prev,"task1_oracle":t1_orc,"task1_auc":t1_auc,
               "task1_weights":{NAMES[i]:float(w[i]) for i in range(len(NAMES))},
               "task2_ndcg_skncp":r_sk['ndcg@3'],"task2_ndcg_skncp_adctr":r_skad['ndcg@3'],"task2_ndcg_rawcos":r_cos['ndcg@3'],
               "leak_audit":"sorted 80/20 internal for Task1 selection; SKNCP index internal-train for iva, full-train for ext/Task2; external answers final scoring only"},
              open("/tmp/pumasi_res/koskncp.json","w"))
    print(f"\nTASK1_F1={t1_prev:.4f}  TASK2_NDCG3={r_skad['ndcg@3']:.4f}  total {time.time()-t0:.0f}s")

if __name__=="__main__": main()
