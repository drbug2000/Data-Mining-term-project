import numpy as np, pandas as pd, sys
from pathlib import Path
ROOT=Path("/home/gyuch_an02/marijuana/Data-Mining-term-project");sys.path.insert(0,str(ROOT))
D=ROOT/"../datasets"
tr=pd.read_csv(D/"search_stream_training.csv");si=pd.read_csv(D/"searchinfo.csv")
vq=pd.read_csv(D/"click_validation_query.csv");va=pd.read_csv(D/"click_validation_answer.csv")
def auc(s,y):
    s=np.asarray(s);y=np.asarray(y);p=s[y==1];n=s[y==0]
    if len(p)==0 or len(n)==0: return .5
    import numpy as _n; return float((_n.sum(p[:,None]>n[None,:])+0.5*_n.sum(p[:,None]==n[None,:]))/(len(p)*len(n)))
o=open("/tmp/eda.txt","w")
def P(s): o.write(s+"\n")
P(f"[규모] train {len(tr)}행 클릭 {int(tr.IsClick.sum())} CTR {tr.IsClick.mean():.4%} | 외부 {len(va)}행 클릭 {int(va.IsClick.sum())} CTR {va.IsClick.mean():.4%}")
P(f"[SearchID 겹침] train∩외부 = {len(set(tr.SearchID)&set(vq.SearchID))} / 외부고유 {vq.SearchID.nunique()}  (≈{len(set(tr.SearchID)&set(vq.SearchID))/vq.SearchID.nunique():.2%})")
# 미관측 유저
s2u=dict(zip(si.SearchID,si.UserID)); tru=set(tr.SearchID.map(s2u));
vu=vq.SearchID.map(s2u); P(f"[미관측 유저] 외부 유저 중 train 미등장 = {(~vu.isin(tru)).mean():.1%}")
# HistCTR 구간별 CTR + AUC
m=tr.merge(va[['SearchID','AdID']],on=['SearchID','AdID'],how='left') if False else tr
bins=[0,0.005,0.02,0.05,0.1,1]; tr['hb']=pd.cut(tr.HistCTR,bins)
P("[HistCTR 구간별 CTR] "+", ".join(f"{str(i)}:{g.IsClick.mean():.3%}" for i,g in tr.groupby('hb',observed=True)))
P(f"[HistCTR 단독 AUC] {auc(tr.HistCTR.values,tr.IsClick.values):.4f}")
# Position
P("[Position별 CTR] "+", ".join(f"p{int(p)}:{g.IsClick.mean():.3%}" for p,g in tr.groupby('Position') if len(g)>100))
# login
si_l=dict(zip(si.SearchID,si.IsUserLoggedOn)); tr['lg']=tr.SearchID.map(si_l)
cl=tr.groupby('lg').IsClick.mean(); P(f"[로그인] 비로그인 {cl.get(0,float('nan')):.3%} vs 로그인 {cl.get(1,float('nan')):.3%} (비율 {cl.get(0,1)/max(cl.get(1,1e-9),1e-9):.2f})")
# ads per search
aps=tr.groupby('SearchID').size().value_counts(normalize=True).sort_index()
P("[검색당 광고수] "+", ".join(f"{k}개:{v:.0%}" for k,v in aps.head(4).items()))
# 직접 의미 cosine AUC (샘플)
SE=np.load(D/"searchinfo_text_embs.npy").astype(np.float32);AD=np.load(D/"adinfo_title_embs.npy").astype(np.float32)
adi=pd.read_csv(D/"adinfo.csv");s2r={int(s):i for i,s in enumerate(si.SearchID)};a2r={int(a):i for i,a in enumerate(adi.AdID)}
samp=tr.sample(30000,random_state=0)
def L2(x):return x/(np.linalg.norm(x)+1e-8)
cos=np.array([float(L2(SE[s2r[int(r.SearchID)]])@L2(AD[a2r[int(r.AdID)]])) if int(r.SearchID) in s2r and int(r.AdID) in a2r else 0. for r in samp.itertuples()])
P(f"[직접 의미 cosine 단독 AUC] {auc(cos,samp.IsClick.values):.4f}  (Cohen's d≈ {(cos[samp.IsClick.values==1].mean()-cos[samp.IsClick.values==0].mean())/cos.std():.3f})")
o.close();print(open("/tmp/eda.txt").read())
