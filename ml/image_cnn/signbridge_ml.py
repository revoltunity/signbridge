#!/usr/bin/env python3
"""SignBridge static-image classification pipeline. Dataset is read-only; outputs go to --work."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, shutil, sys, time
from contextlib import nullcontext
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
SEED=42

def jwrite(path:Path, obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8'); tmp.replace(path)

def sha256(path:Path, chunk=1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()

def discover(root:Path):
    if not root.is_dir(): raise SystemExit(f'Dataset folder tidak ditemukan: {root}')
    classes=sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if len(classes)<2: raise SystemExit('Dibutuhkan minimal 2 subfolder kelas.')
    for cls in classes:
        for p in cls.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS: yield cls.name,p

def audit(args):
    from PIL import Image
    root=Path(args.data).resolve(); work=Path(args.work); report=work/'audit_report.json'
    counts=Counter(); bad=[]; hashes=defaultdict(list); total=0
    for label,p in discover(root):
        total+=1; counts[label]+=1
        try:
            with Image.open(p) as im: im.verify()
            hashes[sha256(p)].append((label,str(p.relative_to(root))))
        except Exception as e: bad.append({'path':str(p.relative_to(root)),'error':str(e)[:300]})
        if total%1000==0: print(f'Diperiksa: {total:,}',flush=True)
    duplicates=[]; cross=[]
    for digest,items in hashes.items():
        if len(items)>1:
            row={'sha256':digest,'files':[{'class':a,'path':b} for a,b in items]}; duplicates.append(row)
            if len({x[0] for x in items})>1: cross.append(row)
    result={'dataset':str(root),'total_images':total,'valid_images':sum(counts.values())-len(bad),'class_counts':dict(sorted(counts.items())),'corrupt_count':len(bad),'corrupt_files':bad,'duplicate_groups':duplicates,'cross_class_duplicate_groups':cross,'note':'Hash SHA256 mendeteksi duplikat byte-identik, bukan near-duplicate.'}
    jwrite(report,result); print(json.dumps({k:v for k,v in result.items() if k not in ('corrupt_files','duplicate_groups','cross_class_duplicate_groups')},indent=2)); print('Laporan:',report)

def split(args):
    from PIL import Image
    root=Path(args.data).resolve(); work=Path(args.work); work.mkdir(parents=True,exist_ok=True)
    # Same byte content is grouped. Identical content assigned to different labels is excluded.
    by_hash=defaultdict(list); bad=[]
    for label,p in discover(root):
        try:
            with Image.open(p) as im: im.verify()
            by_hash[sha256(p)].append((label,p))
        except Exception as e: bad.append({'path':str(p.relative_to(root)),'error':str(e)[:200]})
    cross={h for h,items in by_hash.items() if len({x[0] for x in items})>1}
    groups=defaultdict(list)
    for h,items in by_hash.items():
        if h in cross: continue
        label=items[0][0]
        groups[label].append((h,items))
    rng=random.Random(args.seed); rows=[]; mapping={}; summary={}
    for idx,label in enumerate(sorted(groups)):
        mapping[str(idx)]=label
        gs=groups[label]; rng.shuffle(gs); n=len(gs)
        ntr=max(1,round(n*args.train_ratio)); nva=max(1,round(n*args.val_ratio)) if n>=3 else (1 if n==2 else 0)
        if ntr+nva>=n and n>=3: ntr=max(1,n-2); nva=1
        assignments=['train']*ntr+['val']*nva+['test']*(n-ntr-nva)
        for (h,items),part in zip(gs,assignments):
            for lab,p in items: rows.append({'path':p.relative_to(root).as_posix(),'label':lab,'class_index':idx,'split':part,'group_sha256':h})
        summary[label]={'unique_groups':n,'train_groups':assignments.count('train'),'val_groups':assignments.count('val'),'test_groups':assignments.count('test')}
    if not rows: raise SystemExit('Tidak ada gambar valid setelah audit/cross-label duplicate exclusion.')
    # Write manifest atomically; dataset itself is never copied or modified.
    with (work/'manifest.csv.tmp').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (work/'manifest.csv.tmp').replace(work/'manifest.csv')
    jwrite(work/'class_mapping.json',{'index_to_class':mapping,'class_to_index':{v:int(k) for k,v in mapping.items()}})
    jwrite(work/'split_summary.json',{'seed':args.seed,'ratios':{'train':args.train_ratio,'val':args.val_ratio,'test':args.test_ratio},'classes':summary,'excluded_cross_class_duplicate_groups':len(cross),'corrupt_files':bad,'grouping':'exact SHA256 only'})
    print(f"Manifest: {work/'manifest.csv'} | gambar: {len(rows):,} | kelas: {len(mapping)} | grup duplikat lintas kelas dikecualikan: {len(cross)}")

def deps():
    try:
        import torch, torchvision, PIL, sklearn
        return torch,torchvision
    except ImportError as e: raise SystemExit(f'Dependensi belum terpasang: {e}. Jalankan pip install -r requirements.txt')

def load_rows(work, split_name):
    with (work/'manifest.csv').open(newline='',encoding='utf-8') as f: return [r for r in csv.DictReader(f) if r['split']==split_name]

def train(args):
    torch, tv=deps(); from PIL import Image
    from torch.utils.data import Dataset,DataLoader
    from torchvision import transforms
    work=Path(args.work); rows_train=load_rows(work,'train'); rows_val=load_rows(work,'val')
    if not rows_train or not rows_val: raise SystemExit('Manifest train/val kosong. Jalankan split dengan dataset memadai.')
    mapping=json.loads((work/'class_mapping.json').read_text(encoding='utf-8'))['class_to_index']; root=Path(args.data).resolve()
    class DS(Dataset):
        def __init__(self,rows,tf): self.rows=rows; self.tf=tf
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            r=self.rows[i]
            with Image.open(root/r['path']) as im: im=im.convert('RGB'); x=self.tf(im)
            return x,int(r['class_index'])
    tf_train=transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    tf_val=transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'); print('Device:',device)
    train_dl=DataLoader(DS(rows_train,tf_train),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    val_dl=DataLoader(DS(rows_val,tf_val),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    weights=tv.models.MobileNet_V3_Small_Weights.DEFAULT if not args.no_pretrained else None
    model=tv.models.mobilenet_v3_small(weights=weights); model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,len(mapping)); model.to(device)
    criterion=torch.nn.CrossEntropyLoss(); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    # PyTorch 2.1 compatibility: use the CUDA AMP namespace only on CUDA.
    # On CPU, scaler stays None and autocast is a no-op context.
    amp_enabled = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    start=1; best=0.; ckpt=work/'checkpoint_last.pt'; best_path=work/'model_best.pt'
    if args.resume and ckpt.exists():
        state=torch.load(ckpt,map_location=device,weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scaler is not None and state.get('scaler'):
            scaler.load_state_dict(state['scaler'])
        start=state['epoch']+1; best=state.get('best_val_acc',0.); print('Resume dari epoch',start)
    for epoch in range(start,args.epochs+1):
        model.train(); loss_sum=correct=seen=0
        for x,y in train_dl:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            amp_context = torch.cuda.amp.autocast() if amp_enabled else nullcontext()
            with amp_context:
                out=model(x); loss=criterion(out,y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard CPU training: forward -> loss -> backward -> optimizer.
                loss.backward()
                optimizer.step()
            loss_sum+=loss.item()*len(y); correct+=(out.argmax(1)==y).sum().item(); seen+=len(y)
        model.eval(); vc=vn=0; vloss=0.
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device); out=model(x); vloss+=criterion(out,y).item()*len(y); vc+=(out.argmax(1)==y).sum().item(); vn+=len(y)
        vacc=vc/max(vn,1); print(f"Epoch {epoch}/{args.epochs} loss={loss_sum/max(seen,1):.4f} train_acc={correct/max(seen,1):.4f} val_loss={vloss/max(vn,1):.4f} val_acc={vacc:.4f}",flush=True)
        state={'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict() if scaler is not None else None,'best_val_acc':max(best,vacc),'class_to_index':mapping,'image_size':args.image_size}
        torch.save(state,ckpt)
        if vacc>best:
            best=vacc; torch.save({'model':model.state_dict(),'class_to_index':mapping,'image_size':args.image_size,'architecture':'mobilenet_v3_small','best_val_accuracy':best},best_path)
    jwrite(work/'preprocessing.json',{'image_size':args.image_size,'color_mode':'RGB','resize':'direct square resize','normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},'inference':'same resize and normalization'})
    print('Checkpoint:',ckpt,'\nModel terbaik:',best_path)

def evaluate(args):
    torch,tv=deps(); import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset,DataLoader
    from torchvision import transforms
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    work=Path(args.work); ckpt=torch.load(args.checkpoint or work/'model_best.pt',map_location='cpu',weights_only=False)
    mapping=ckpt['class_to_index']; idx_to={int(v):k for k,v in mapping.items()}; root=Path(args.data).resolve(); rows=load_rows(work,'test')
    if not rows: raise SystemExit('Test split kosong.')
    size=int(ckpt.get('image_size',224)); tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    class DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self,i):
            r=rows[i]
            with Image.open(root/r['path']) as im: x=tf(im.convert('RGB'))
            return x,int(r['class_index'])
    dl=DataLoader(DS(),batch_size=args.batch_size,num_workers=args.workers)
    model=tv.models.mobilenet_v3_small(weights=None); model.classifier[-1]=torch.nn.Linear(model.classifier[-1].in_features,len(mapping)); model.load_state_dict(ckpt['model']); model.eval()
    ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for x,y in dl:
            out=model(x); pr=torch.softmax(out,1); ys.extend(y.numpy().tolist()); ps.extend(pr.argmax(1).numpy().tolist()); probs.extend(pr.max(1).values.numpy().tolist())
    labels=list(range(len(mapping))); names=[idx_to[i] for i in labels]
    report=classification_report(ys,ps,labels=labels,target_names=names,output_dict=True,zero_division=0)
    result={'accuracy':accuracy_score(ys,ps),'classification_report':report,'confusion_matrix':confusion_matrix(ys,ps,labels=labels).tolist(),'class_order':names,'test_samples':len(ys),'mean_max_probability':float(np.mean(probs))}
    jwrite(work/'evaluation_report.json',result); print(json.dumps({'accuracy':result['accuracy'],'test_samples':len(ys),'report':report},indent=2))

def parser():
    p=argparse.ArgumentParser(description='SignBridge ML pipeline'); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('audit'); a.add_argument('--data',required=True); a.add_argument('--work',default='signbridge_work'); a.set_defaults(fn=audit)
    s=sub.add_parser('split'); s.add_argument('--data',required=True); s.add_argument('--work',default='signbridge_work'); s.add_argument('--seed',type=int,default=SEED); s.add_argument('--train-ratio',type=float,default=.70); s.add_argument('--val-ratio',type=float,default=.15); s.add_argument('--test-ratio',type=float,default=.15); s.set_defaults(fn=split)
    t=sub.add_parser('train'); t.add_argument('--data',required=True); t.add_argument('--work',default='signbridge_work'); t.add_argument('--epochs',type=int,default=20); t.add_argument('--batch-size',type=int,default=32); t.add_argument('--workers',type=int,default=4); t.add_argument('--image-size',type=int,default=224); t.add_argument('--lr',type=float,default=1e-4); t.add_argument('--resume',action='store_true'); t.add_argument('--cpu',action='store_true'); t.add_argument('--no-pretrained',action='store_true'); t.set_defaults(fn=train)
    e=sub.add_parser('evaluate'); e.add_argument('--data',required=True); e.add_argument('--work',default='signbridge_work'); e.add_argument('--checkpoint'); e.add_argument('--batch-size',type=int,default=32); e.add_argument('--workers',type=int,default=2); e.set_defaults(fn=evaluate)
    return p
if __name__=='__main__':
    args=parser().parse_args()
    if args.cmd=='split' and abs(args.train_ratio+args.val_ratio+args.test_ratio-1)>1e-6: raise SystemExit('Rasio split harus berjumlah 1.0')
    args.fn(args)
