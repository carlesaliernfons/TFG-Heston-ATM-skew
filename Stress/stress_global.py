"""Global Heston calibration (vega approximation, as in calibration_by_maturity.ipynb), market and model ATM skew, power-law fits and decomposition-formula error
on a calm day and three stressed days. Writes g_global.csv, g_skew.csv, g_power.csv and g_decomp.csv (about 4 minutes)."""
import os, numpy as np, pandas as pd, time
import heston_tfg as ht
from stress_lib import load_day_clean
DAYS=["2026-10-12","2026-11-13","2026-11-19","2026-11-20"]
glob=[];sk=[];pl=[];dec=[]
for day in DAYS:
    r,chain,q=load_day_clean(day); cal=ht.thin_chain(chain,max_per_T=40)
    mk=pd.read_csv(os.path.join("..", "Data", day, "market.csv")).iloc[0]
    ms=ht.market_atm_skew(chain)
    for lab,m in [("all",ms["days"]>0),("ge14",ms["days"]>=14)]:
        f=ht.fit_power_law(ms["T"][m],ms["atm_skew"][m]); pl.append(dict(day=day,sample=lab,**f))
    v=float(np.clip(cal.loc[cal["T"].idxmin(),"iv"]**2,0.001,0.5))
    starts=[[10.97,0.0372,1.865,-0.59,0.008],[5.0,0.05,1.5,-0.7,v],[15,0.03,3.0,-0.6,v]]
    t0=time.time(); p,fit,rmse=ht.calibrate(cal,r,"fourier",starts=starts,verbose=False)
    fit["err"]=fit["model_iv"]-fit["iv"]
    byT=fit.groupby("T")["err"].apply(lambda e: 100*np.sqrt(np.mean(e**2)))
    glob.append(dict(day=day,VIX=mk["VIX"],spot=mk["spot_SPX"],n=len(cal),nT=cal["T"].nunique(),**p,rmse=100*rmse,
        phi=2*p["kappa"]*p["theta"]/p["nu"]**2,limit=ht.heston_short_skew_limit(p),err_short=byT.iloc[0],err_maxlong=byT[byT.index>=30/365].max(),secs=time.time()-t0))
    print(glob[-1],flush=True)
    f=ht.model_atm_skew(p,ms["T"].values,"fourier"); d=ht.model_atm_skew(p,ms["T"].values,"decomposition")
    TT=np.array([1,2,4.5,10,21,44,94,199])/365
    fr=ht.model_atm_skew(p,TT,"fourier",h=1e-4)["atm_skew"].values; dr=ht.model_atm_skew(p,TT,"decomposition",h=1e-4)["atm_skew"].values
    for t,a,b in zip(TT,fr,dr): dec.append(dict(day=day,days=t*365,fourier=a,decomp=b,rel=b/a-1))
    pd.DataFrame(dec).to_csv("g_decomp.csv",index=False)
    for i,row in ms.reset_index(drop=True).iterrows():
        sk.append(dict(day=day,days=row["days"],n=row["n"],mkt=row["atm_skew"],se=row["atm_skew_se"],atm_iv=row["atm_iv"],heston=f["atm_skew"].iloc[i],decomp=d["atm_skew"].iloc[i]))
    pd.DataFrame(glob).to_csv("g_global.csv",index=False); pd.DataFrame(sk).to_csv("g_skew.csv",index=False); pd.DataFrame(pl).to_csv("g_power.csv",index=False)
print("END")
