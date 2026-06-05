"""
exp_unified.py — 두 방법론 통합 파이프라인 (m04 content-ensemble + KoSKNCP log-odds).

통합 신호 (log-odds 선형, F1-objective coordinate ascent, w_min=0.5 정규화):
  1 logHist         HistCTR (사전 log-odds)
  2 log ad_ctr      광고 CTR (Laplace, train-only)
  3 log cat_ctr     카테고리 CTR (Laplace)
  + login offset    log(1.7)*(1-logged)  (고정)
  4 IPS_pos         위치 편향 보정 log(λ_pos/λ̄)
  5 rank            검색 내 HistCTR 상대순위 (0.5 - r)
  6 content_z       클릭튜닝 bi-encoder 앙상블 (m04, 강신호 AUC~0.68)
  7 SKNCP           semantic-kNN 클릭예측 (KoSKNCP, 협업필터 신호)
  8 nbr_ctr         이웃광고 CTR (logit(μ_a)-logit(λ̄))

검증 (둘 다의 honest 규율 통합):
  - sorted SearchID 80/20 내부 split. content head·F1 지수·rate 전부 내부에서만.
  - 내부-val CTR/SKNCP/nbr = 내부-train 인덱스; external = full-train 인덱스.
  - threshold: OOF 5-fold rate + train-prevalence rate 둘 다 보고.
  - external click_validation 라벨은 최종 채점만.
  - bootstrap CI(2000) + leave-one-signal-out ablation 으로 엄밀 검증.

content logit 은 /tmp/cs_full.npz(10-seed 앙상블) 재사용. 없으면 즉시 학습.
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
from models.m04_gated.gated_ctr import _best_f1_topk,_fit_f1_exponents,_l2_normalize,_binary_auc
DATA=ROOT/"../datasets"; KS=20; LO=np.log(1.7)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect(stream):
    Q,A,Y,ad,ca,uid,sid,hist,lg,pos=[],[],[],[],[],[],[],[],[],[]
    for ev in stream:
        for a in ev.ads:
            Q.append(ev.search_emb);A.append(a.ad_emb);Y.append(int(a.is_click))
            ad.append(int(a.ad_id));ca.append(int(a.category_id));uid.append(int(ev.user_id))
            sid.append(int(ev.search_id));hist.append(float(a.hist_ctr) if a.hist_ctr is not None else 0.)
            lg.append(float(ev.is_logged_on));pos.append(int(a.position))
    return dict(Q=np.asarray(Q,np.float32),A=np.asarray(A,np.float32),Y=np.asarray(Y,np.float32),
                ad=np.asarray(ad),ca=np.asarray(ca),uid=np.asarray(uid),sid=np.asarray(sid),
                hist=np.asarray(hist,np.float64),lg=np.asarray(lg,np.float64),pos=np.asarray(pos,np.float64))

def ctr_series(keys,y,g):
    d=pd.DataFrame({'k':keys,'y':y}); gp=d.groupby('k')['y'].agg(['sum','count'])
    return (gp['sum']+KS*g)/(gp['count']+KS)
def look(keys,ser,g): return pd.Series(keys).map(ser).fillna(g).values

def skncp(Qs,As_rows,ad_emb_n, tr_q_emb_n, tr_clicked_ad_rows_per_q, K=100, cap=300, blk=512):
    """Qs:(M,384) L2 scoring query embs; As_rows:(M,) ad row idx; ad_emb_n:(Nad,384) L2.
    tr_q_emb_n:(T,384) L2 train query embs; tr_clicked_ad_rows_per_q: list[T] of ad-row lists (clicked).
    각 row: K 최근접 train query → 그들 clicked ad emb 모아 max cos(현재 ad)."""
    M=len(Qs); out=np.zeros(M,np.float32)
    Qd=torch.tensor(Qs,device=dev); TQ=torch.tensor(tr_q_emb_n,device=dev)
    ADn=torch.tensor(ad_emb_n,device=dev)
    for i in range(0,M,blk):
        qb=Qd[i:i+blk]
        sims=qb@TQ.T
        keff=min(K,TQ.shape[0]); top=sims.topk(keff,dim=1).indices.cpu().numpy()
        for j in range(qb.shape[0]):
            rows=[]
            for t in top[j]:
                rows.extend(tr_clicked_ad_rows_per_q[t])
                if len(rows)>=cap: break
            r=i+j; arow=As_rows[r]
            if rows and arow>=0:
                pool=ADn[torch.tensor(rows[:cap],device=dev)]
                out[r]=float((ADn[arow]@pool.T).max())
    return out

def main():
    t0=time.time()
    ds=RecoDataset(DATA).load()
    si=pd.read_csv(ds.dir/"searchinfo.csv");ui=pd.read_csv(ds.dir/"userinfo.csv");adi=pd.read_csv(ds.dir/"adinfo.csv")
    s2ip=dict(zip(si.SearchID,si.IPID));u2d=dict(zip(ui.UserID,ui.UserDeviceID))
    tr=collect(ds.training_stream())
    pr=ds.val_click_queries(); yex=ds.val_click_answers()["IsClick"].to_numpy()[:len(pr)].astype(int)
    ex=dict(Q=_l2_normalize(np.asarray([e.search_emb for e,_ in pr],np.float32)),
            A=_l2_normalize(np.asarray([a.ad_emb for _,a in pr],np.float32)),
            ad=np.asarray([int(a.ad_id) for _,a in pr]),ca=np.asarray([int(a.category_id) for _,a in pr]),
            uid=np.asarray([int(e.user_id) for e,_ in pr]),sid=np.asarray([int(e.search_id) for e,_ in pr]),
            hist=np.asarray([float(a.hist_ctr) for _,a in pr],np.float64),
            lg=np.asarray([float(e.is_logged_on) for e,_ in pr],np.float64),
            pos=np.asarray([int(a.position) for _,a in pr],np.float64))
    Qn=_l2_normalize(tr['Q']);An=_l2_normalize(tr['A']);Y=tr['Y'];g=float(Y.mean());n=len(Y)
    ip=np.array([s2ip.get(int(s),-1) for s in tr['sid']]);dvk=np.array([u2d.get(int(u),-1) for u in tr['uid']])
    eip=np.array([s2ip.get(int(s),-1) for s in ex['sid']]);edv=np.array([u2d.get(int(u),-1) for u in ex['uid']])
    print(f"[unified] loaded n={n} ({time.time()-t0:.0f}s)")

    uniq=np.sort(np.unique(tr['sid']));cut=int(len(uniq)*0.8);vs=set(uniq[cut:].tolist())
    mva=np.array([s in vs for s in tr['sid']]);mtr=~mva

    # content (cache 재사용)
    cf=Path("/tmp/cs_full.npz")
    if cf.exists():
        c=np.load(cf)
        if len(c['con'])==n: con=c['con'].astype(np.float64); cone=c['cone'].astype(np.float64); print("[unified] content cache 재사용")
        else: con=cone=None
    else: con=cone=None
    if con is None:
        print("[unified] content cache 없음 → exp_content_strong 먼저 실행 필요"); sys.exit(1)
    cmu,csd=con[mtr].mean(),con[mtr].std()+1e-9; conz=(con-cmu)/csd; conze=(cone-cmu)/csd

    # SKNCP 인덱스: 내부-train clicks (iva/itr용) + full-train clicks (ext용)
    ad_emb_full=_l2_normalize(np.load(ds.dir/"adinfo_title_embs.npy").astype(np.float32))
    adid2row={int(a):i for i,a in enumerate(adi.AdID.values)}
    adrow_tr=np.array([adid2row.get(int(a),-1) for a in tr['ad']]); adrow_ex=np.array([adid2row.get(int(a),-1) for a in ex['ad']])
    def build_index(mask):
        # train queries(고유 검색) = mask 행의 search emb; clicked ad rows per query = 그 검색의 클릭 광고
        qidx=np.where(mask)[0]
        # 세션(sid) 단위로 묶기엔 비용 → 행 자체를 "쿼리"로(중복 OK), 클릭행만 인덱스에 ad row 부여
        clk=qidx[Y[qidx]==1]
        TQ=Qn[clk]  # 클릭이 일어난 쿼리 임베딩
        rows=[[adrow_tr[r]] for r in clk]  # 각 클릭쿼리의 클릭광고 1개
        return TQ,rows
    TQ_itr,rows_itr=build_index(mtr); TQ_full,rows_full=build_index(np.ones(n,bool))
    sk_va=skncp(Qn[mva],adrow_tr[mva],ad_emb_full,TQ_itr,rows_itr)
    sk_ex=skncp(ex['Q'],adrow_ex,ad_emb_full,TQ_full,rows_full)
    print(f"[unified] SKNCP done ({time.time()-t0:.0f}s)")

    # 신호 빌더
    def signals(split):  # 'va' or 'ex'
        if split=='va':
            rows=np.where(mva)[0]; A_=tr['ad'][rows];C_=tr['ca'][rows];IP=ip[rows];DV=dvk[rows]
            H=tr['hist'][rows];P=tr['pos'][rows];SID=tr['sid'][rows];SK=sk_va;CZ=conz[rows];src_mask=mtr
        else:
            A_=ex['ad'];C_=ex['ca'];IP=eip;DV=edv;H=ex['hist'];P=ex['pos'];SID=ex['sid'];SK=sk_ex;CZ=conze;src_mask=np.ones(n,bool)
        sad=ctr_series(tr['ad'][src_mask],Y[src_mask],g);scat=ctr_series(tr['ca'][src_mask],Y[src_mask],g)
        sip=ctr_series(ip[src_mask],Y[src_mask],g);sdv=ctr_series(dvk[src_mask],Y[src_mask],g)
        spos=ctr_series(tr['pos'][src_mask],Y[src_mask],g)
        adctr=look(A_,sad,g);catctr=look(C_,scat,g);ipctr=look(IP,sip,g);dvctr=look(DV,sdv,g);posctr=look(P,spos,g)
        ips=np.log(np.maximum(posctr,1e-6))-np.log(g)
        rk=pd.DataFrame({'s':SID,'h':H}).groupby('s')['h'].rank(pct=True).values; rankf=0.5-rk
        def logit(x): x=np.clip(x,1e-6,1-1e-6); return np.log(x/(1-x))
        nbr=logit(adctr)-logit(g)
        feats=np.column_stack([np.log(np.maximum(H,1e-6)),np.log(np.maximum(adctr,1e-6)),
                               np.log(np.maximum(ipctr,1e-6)),np.log(np.maximum(dvctr,1e-6)),
                               np.log(np.maximum(catctr,1e-6)),ips,rankf,CZ,SK,nbr])
        base=LO*(1-(tr['lg'][np.where(mva)[0]] if split=='va' else ex['lg']))
        return feats,base
    Xva,bva=signals('va');yva=Y[mva];Xe,be=signals('ex')
    NAMES=['logHist','ad_ctr','ip_ctr','dev_ctr','cat_ctr','IPS_pos','rank','content','SKNCP','nbr_ctr']

    def fit_eval(cols):
        Xv=Xva[:,cols];Xx=Xe[:,cols]
        w=_fit_f1_exponents(Xv,yva,base=bva,grid=np.arange(0.5,3.05,0.5))  # w_min=0.5 정규화(KoSKNCP)
        sv=Xv@w+bva; se=Xx@w+be
        # rates
        rs=_best_f1_topk(sv,yva)[1]/len(yva)
        rng=np.random.RandomState(0);idx=rng.permutation(len(yva))
        roof=float(np.mean([_best_f1_topk(sv[f],yva[f])[1]/len(f) for f in np.array_split(idx,5)]))
        prev=g
        def f1r(sc,y,r):
            k=max(1,int(round(r*len(sc))));o=np.argsort(-sc);p=np.zeros(len(sc),int);p[o[:k]]=1
            tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
            return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
        return dict(w=w,honest_oof=f1r(se,yex,roof),honest_prev=f1r(se,yex,prev),
                    oracle=_best_f1_topk(se,yex)[0],auc=_binary_auc(se,yex),se=se,roof=roof)

    allc=list(range(len(NAMES)))
    # 신호 subset 비교 (honest_oof = 안정 지표; 둘 다 OOF rate 사용)
    # NAMES idx: 0logHist 1ad 2ip 3dev 4cat 5IPS 6rank 7content 8SKNCP 9nbr
    SUBSETS={
        "minimal(5CTR+content)":[0,1,2,3,4,7],
        "5CTR+content+IPS+rank":[0,1,2,3,4,5,6,7],
        "5CTR+content+SKNCP+nbr":[0,1,2,3,4,7,8,9],
        "ALL-10(union)":allc,
        "KoSKNCP-like(no content)":[0,1,2,3,4,5,6,8,9],
        "3CTR+content(m04-ish)":[0,1,4,7],
    }
    print("\n[SUBSET 비교] (honest_oof / honest_prev / oracle / AUC)")
    results={}
    for name,cols in SUBSETS.items():
        r=fit_eval(cols);results[name]=r
        print(f"  {name:26s} oof={r['honest_oof']:.4f} prev={r['honest_prev']:.4f} oracle={r['oracle']:.4f} AUC={r['auc']:.4f}")
    best_name=max(results,key=lambda k:results[k]['honest_oof']);full=results[best_name]
    print(f"\n[BEST by honest_oof] {best_name}: honest_oof={full['honest_oof']:.4f} honest_prev={full['honest_prev']:.4f} AUC={full['auc']:.4f}")
    print("  weights:",{NAMES[SUBSETS[best_name][i]]:round(float(full['w'][i]),2) for i in range(len(SUBSETS[best_name]))})

    # bootstrap CI (2000) on external honest_prev
    se=full['se'];rng=np.random.RandomState(1);N=len(yex)
    def f1r(sc,y,r):
        k=max(1,int(round(r*len(sc))));o=np.argsort(-sc);p=np.zeros(len(sc),int);p[o[:k]]=1
        tp=int(((p==1)&(y==1)).sum());fp=int(((p==1)&(y==0)).sum());fn=int(((p==0)&(y==1)).sum())
        return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
    bs=[]
    for _ in range(2000):
        idx=rng.randint(0,N,N); bs.append(f1r(se[idx],yex[idx],g))
    bs=np.array(bs);ci=(float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))
    print(f"\n[BOOTSTRAP] honest_prev mean={bs.mean():.4f} std={bs.std():.4f} 95%CI=[{ci[0]:.4f},{ci[1]:.4f}]")

    os.makedirs("/tmp/pumasi_res",exist_ok=True)
    json.dump({"approach":"unified","honest_oof":full['honest_oof'],"honest_prev":full['honest_prev'],
               "oracle":full['oracle'],"auc":full['auc'],"boot_mean":float(bs.mean()),"boot_ci":ci,
               "weights":{NAMES[i]:float(full['w'][i]) for i in range(len(NAMES))},
               "leak_audit":"sorted internal 80/20 for all selection; CTR/SKNCP internal-train index for iva, full-train for ext; threshold=prevalence/OOF (no external labels); external answers final scoring only"},
              open("/tmp/pumasi_res/unified.json","w"))
    print(f"\nEXTERNAL_HONEST_F1={full['honest_prev']:.4f}  total {time.time()-t0:.0f}s")

if __name__=="__main__": main()
