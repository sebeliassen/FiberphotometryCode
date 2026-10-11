#!/usr/bin/env python3
"""Compute counts-only vs counts+dwell split for the presentation ladder."""
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models")
cache = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/analysis/language_models/output/strategies_extended/continuous_bout_cache.pkl"

print(f"[{time.strftime('%H:%M:%S')}] Loading cache...", flush=True)
with open(cache, "rb") as f:
    data = pickle.load(f)
sb = data["session_bouts"] if isinstance(data, dict) and "session_bouts" in data else data

FPS, W, H, STR, K = 80.025, int(68.3*80.025), int(120.0*80.025), int(5.0*80.025), 100

def js(P, Q, e=1e-12):
    P = np.clip(P,e,1); Q = np.clip(Q,e,1)
    P /= P.sum(1,keepdims=True); Q /= Q.sum(1,keepdims=True)
    M = .5*(P+Q)
    return float(np.mean(.5*(np.sum(P*np.log2(P/M),1)+np.sum(Q*np.log2(Q/M),1))))

print(f"[{time.strftime('%H:%M:%S')}] Preprocessing...", flush=True)
SS = []
for si, bouts in enumerate(sb):
    if len(bouts) < 100: continue
    N = bouts[-1]["end_frame"]+1
    fs = np.zeros(N, dtype=np.int32)
    sp = np.zeros((N,6), dtype=np.float32)
    for b in bouts:
        sf,ef = b["start_frame"],b["end_frame"]
        sid = min(b["syllable"],99); fs[sf:ef+1] = sid
        v = b.get("vector")
        if v is not None and len(v)>=14:
            sp[sf:ef+1,:4]=v[:4]; sp[sf:ef+1,4]=v[10]; sp[sf:ef+1,5]=v[12]
    d = np.where(fs[1:]!=fs[:-1])[0]+1
    on = np.zeros(N,dtype=bool); on[0]=True; on[d]=True
    oh_b = np.zeros((N,K),dtype=np.float32); oh_b[on,fs[on]]=1.0
    cb = np.vstack([np.zeros((1,K),dtype=np.float32), np.cumsum(oh_b,0)])
    oh_f = np.zeros((N,K),dtype=np.float32); oh_f[np.arange(N),fs]=1.0
    cf = np.vstack([np.zeros((1,K),dtype=np.float32), np.cumsum(oh_f,0)])
    nb = np.zeros(N,dtype=np.int32); c=0
    for dd in d: nb[c:dd]=dd; c=dd
    nb[c:]=N
    cs = np.vstack([np.zeros((1,6),dtype=np.float32), np.cumsum(sp,0)])
    cs2= np.vstack([np.zeros((1,6),dtype=np.float32), np.cumsum(sp**2,0)])
    SS.append({"si":si,"N":N,"fs":fs,"cb":cb,"cf":cf,"nb":nb,"cs":cs,"cs2":cs2})

print(f"[{time.strftime('%H:%M:%S')}] {len(SS)} sessions. Extracting features...", flush=True)

aP3, aCnt, aTime, aTgt, aG = [],[],[],[],[]
for s in SS:
    N=s["N"]
    if N < W+H+200: continue
    ta = np.arange(W,N-H,STR)
    ts = s["nb"][ta]; v=(ts+H<N); ta=ta[v]; ts=ts[v]
    M=len(ta)
    if M==0: continue
    tn=ta/float(N); p1=np.column_stack([tn,tn**2]).astype(np.float32)
    ss=s["cs"][ta]-s["cs"][ta-W]; sm=ss/float(W)
    sq=s["cs2"][ta]-s["cs2"][ta-W]; sd=np.sqrt(np.maximum(sq/float(W)-sm**2,1e-6))
    p2=np.hstack([p1,sm,sd]).astype(np.float32)
    oh=np.zeros((M,K),dtype=np.float32); oh[np.arange(M),s["fs"][ta]]=1.0
    p3=np.hstack([p2,oh]).astype(np.float32)
    bc=s["cb"][ta]-s["cb"][ta-W]; bd=(bc+1e-4)/(bc.sum(1,keepdims=True)+1e-4*K)
    fc=s["cf"][ta]-s["cf"][ta-W]; fd=(fc+1e-4)/(fc.sum(1,keepdims=True)+1e-4*K)
    tg=s["cb"][ts+H]-s["cb"][ts]; tg=(tg+1e-4)/(tg.sum(1,keepdims=True)+1e-4*K)
    aP3.append(p3); aCnt.append(bd); aTime.append(fd); aTgt.append(tg); aG.append(np.full(M,s["si"],dtype=int))

X3=np.vstack(aP3); Xc=np.hstack([X3,np.vstack(aCnt)]); Xf=np.hstack([X3,np.vstack(aCnt),np.vstack(aTime)])
Y=np.vstack(aTgt); G=np.concatenate(aG)
print(f"[{time.strftime('%H:%M:%S')}] {len(Y):,} samples. Running 5-fold CV...", flush=True)

gc,gf=[],[]
for f,(tr,te) in enumerate(GroupKFold(5).split(X3,groups=G)):
    Ytr,Yte=Y[tr],Y[te]
    pg=np.tile(Ytr.mean(0,keepdims=True),(len(te),1)); pg/=pg.sum(1,keepdims=True)
    jb=js(Yte,pg)
    pc=np.clip(Ridge(1.0).fit(Xc[tr],Ytr).predict(Xc[te]),1e-4,1.0); pc/=pc.sum(1,keepdims=True)
    pf=np.clip(Ridge(1.0).fit(Xf[tr],Ytr).predict(Xf[te]),1e-4,1.0); pf/=pf.sum(1,keepdims=True)
    gc.append(((jb-js(Yte,pc))/jb)*100.0)
    gf.append(((jb-js(Yte,pf))/jb)*100.0)
    print(f"  Fold {f+1}/5: Counts={gc[-1]:.2f}% | Counts+Dwell={gf[-1]:.2f}%", flush=True)

ci=lambda a: 1.96*np.std(a)/np.sqrt(5)
print(f"\nCounts only (P3+bout_counts):    +{np.mean(gc):.2f}% ± {ci(gc):.2f}%  Folds={[round(x,2) for x in gc]}")
print(f"Counts+Dwell (P3+bout+frame):    +{np.mean(gf):.2f}% ± {ci(gf):.2f}%  Folds={[round(x,2) for x in gf]}")
