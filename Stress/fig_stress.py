"""Figure of Section 5 (calm and stressed market). Needs g_skew.csv and stress_by_maturity.csv."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z=1.45; plt.rcParams.update({"font.size":8.6*Z,"axes.grid":True,"grid.alpha":0.25,"axes.spines.top":False,"axes.spines.right":False,"legend.fontsize":8*Z,"xtick.labelsize":8*Z,"ytick.labelsize":8*Z})
s=pd.read_csv("g_skew.csv"); b=pd.read_csv("stress_by_maturity.csv")
bad={('2026-11-19',132),('2026-11-19',315),('2026-11-20',3),('2026-11-20',5),('2026-11-20',131)}
b=b[[ (a,c) not in bad for a,c in zip(b.day,b.days)]]
lab={"2026-10-12":"12 Oct (VIX 15.2)","2026-11-13":"13 Nov (VIX 19.3)","2026-11-19":"19 Nov (VIX 28.4)","2026-11-20":"20 Nov (VIX 35.9)"}
col={"2026-10-12":"#2b6cb0","2026-11-13":"#7f9cc0","2026-11-19":"#e3a35a","2026-11-20":"#c0392b"}
W=5.906*0.95*Z; fig,ax=plt.subplots(1,2,figsize=(W,W*0.42))
for d in ["2026-10-12","2026-11-20"]:
    g=s[s.day==d]
    ax[0].plot(g.days,-g.mkt,"o",color=col[d],ms=5,label=lab[d][:6]+", market")
    ax[0].plot(g.days,-g.heston,"-",color=col[d],lw=1.8,label=lab[d][:6]+", Heston")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("days to maturity"); ax[0].set_ylabel(r"$|\psi(T)|$")
ax[0].legend(frameon=False,fontsize=7.2*Z,loc="lower left")
from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
ax[0].yaxis.set_major_locator(FixedLocator([0.3,0.5,1,2,3])); ax[0].yaxis.set_minor_locator(NullLocator()); ax[0].yaxis.set_major_formatter(FormatStrFormatter("%g"))
ax[0].set_xticks([3,10,30,100,300]); ax[0].xaxis.set_major_formatter(FormatStrFormatter("%g")); ax[1].set_xticks([3,10,30,100,300]); ax[1].xaxis.set_major_formatter(FormatStrFormatter("%g"))
for d in lab:
    g=b[b.day==d]; ax[1].plot(g.days,g.nu,"o-",color=col[d],ms=4,lw=1.4,label=lab[d])
ax[1].set_xscale("log"); ax[1].set_xlabel("days to maturity"); ax[1].set_ylabel(r"$\nu$ (case B)")
ax[1].legend(frameon=False,fontsize=7.2*Z)
for a in ax: a.set_xticks([3,10,30,100,300]); a.xaxis.set_major_formatter(FormatStrFormatter("%g")); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(); fig.savefig("fig_stress.png",dpi=220,bbox_inches="tight")
