"""Figura de distribución del split ISIC congelado (Cap 3.2). Números fijos del split
(isic_split.json, seed 42, balanceado). Salida: _master/fig_isic_split.png"""
import matplotlib.pyplot as plt, numpy as np, os
splits=["Train","Val","Test"]; mel=[5226,1119,1120]; ben=[5225,1120,1120]
x=np.arange(3); w=0.6
fig,ax=plt.subplots(figsize=(6,4))
ax.bar(x,ben,w,label="Benigno",color="#4c78a8")
ax.bar(x,mel,w,bottom=ben,label="Maligno",color="#e45756")
for i,(m,b) in enumerate(zip(mel,ben)):
    ax.text(i,(m+b)+120,f"{m+b}",ha="center",fontsize=9,weight="bold")
    ax.text(i,b/2,f"{b}",ha="center",va="center",color="white",fontsize=8)
    ax.text(i,b+m/2,f"{m}",ha="center",va="center",color="white",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f"{s}\n({m+b})" for s,m,b in zip(splits,mel,ben)])
ax.set_ylabel("Imágenes"); ax.set_title("ISIC-DICM-17K: split estratificado 70/15/15 (balanceado 50/50)")
ax.legend(); ax.set_ylim(0,11500); fig.tight_layout()
out=os.path.join(os.path.dirname(__file__),"..","..","_master","fig_isic_split")
fig.savefig(out+".pdf",bbox_inches="tight"); fig.savefig(out+".png",dpi=150,bbox_inches="tight")
print("guardado fig_isic_split.png | total=14930 (7465 mel / 7465 ben)")
