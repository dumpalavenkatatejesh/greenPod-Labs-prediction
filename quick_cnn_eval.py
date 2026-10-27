import os, sys, json, random
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
CLASSES=["blackspot","canker","fresh","grenning"]
torch.set_num_threads(2)
random.seed(42); np.random.seed(42); torch.manual_seed(42)
root=sys.argv[1]; out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
def rows(split):
    return [(str(p),c) for c in CLASSES for p in sorted((Path(root)/split/c).glob("*")) if p.suffix.lower() in [".jpg",".jpeg",".png"]]
class DS(Dataset):
    def __init__(self,rs,tf): self.rs=rs; self.tf=tf
    def __len__(self): return len(self.rs)
    def __getitem__(self,i):
        p,c=self.rs[i]
        with Image.open(p) as im: im=im.convert("RGB")
        return self.tf(im),CLASSES.index(c)
tf=transforms.Compose([transforms.Resize((64,64)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
ev=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(.25),nn.Linear(64,4))
    def forward(self,x): return self.net(x)
tr=rows("train"); te=rows("test")
dl=DataLoader(DS(tr,tf),batch_size=64,shuffle=True,num_workers=0)
tl=DataLoader(DS(te,ev),batch_size=64,shuffle=False,num_workers=0)
model=CNN(); opt=torch.optim.Adam(model.parameters(),lr=.001); lossf=nn.CrossEntropyLoss()
model.train()
for epoch in range(3):
    model.train()
    running=0.0
    for xb,yb in dl:
        opt.zero_grad(); loss=lossf(model(xb),yb); loss.backward(); opt.step()
        running += float(loss.item())
    print(f"epoch {epoch+1}/3 loss={running/max(1,len(dl)):.4f}")

model.eval(); yt=[]; yp=[]
with torch.no_grad():
    for xb,yb in tl: yt.extend(yb.numpy()); yp.extend(model(xb).argmax(1).numpy())
metrics={"accuracy":float(accuracy_score(yt,yp))}
pr,re,f1,_=precision_recall_fscore_support(yt,yp,average="macro",zero_division=0)
metrics.update({"macro_precision":float(pr),"macro_recall":float(re),"macro_f1":float(f1)})
rep=classification_report(yt,yp,labels=list(range(4)),target_names=CLASSES,zero_division=0,output_dict=True)
json.dump({"metrics":metrics,"classification_report":rep,"note":"Three-epoch quick baseline; not fully tuned. Compare cautiously against the SVM."},open(out/"quick_cnn_three_epoch.json","w"),indent=2)
cm=confusion_matrix(yt,yp,labels=list(range(4))); disp=ConfusionMatrixDisplay(cm,display_labels=CLASSES)
fig,ax=plt.subplots(figsize=(6,5)); disp.plot(ax=ax,xticks_rotation=35,colorbar=False); plt.tight_layout(); plt.savefig(out/"quick_cnn_three_epoch_confusion.png",dpi=130); plt.close()
torch.save(model.state_dict(),out/"quick_cnn_three_epoch.pt")
print(json.dumps(metrics))
