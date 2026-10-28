"""Create deterministic train/validation splits from four class folders.

Input:
  data/raw/pothole/       positive pothole examples
  data/raw/not_pothole/   normal/non-pothole examples
  data/raw/garbage/       positive garbage examples
  data/raw/not_garbage/   clean/non-garbage examples
Output matches train.py's expected ImageFolder layout.
"""
from pathlib import Path
import argparse, random, shutil
from PIL import Image
EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}

def valid_images(folder):
    out=[]
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS:
            try:
                with Image.open(p) as im: im.verify()
                out.append(p)
            except Exception: pass
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',default='data/raw');ap.add_argument('--out',default='data');ap.add_argument('--val-ratio',type=float,default=.2);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--copy',action='store_true',help='Copy instead of hard-linking files')
    a=ap.parse_args(); raw=Path(a.raw); out=Path(a.out); rng=random.Random(a.seed)
    classes={'pothole':['pothole','not_pothole'],'garbage':['garbage','not_garbage']}
    counts={}
    for task,names in classes.items():
        for name in names:
            src=raw/name
            if not src.exists(): raise FileNotFoundError(f'Missing {src}. Put the extracted class images there.')
            files=valid_images(src)
            if len(files)<5: raise ValueError(f'{src} has only {len(files)} valid images; add at least 5 per class.')
            rng.shuffle(files); nval=max(1,round(len(files)*a.val_ratio)); splits={'val':files[:nval],'train':files[nval:]}
            for split,items in splits.items():
                dest=out/task/split/name;dest.mkdir(parents=True,exist_ok=True)
                for i,f in enumerate(items):
                    target=dest/f'{f.stem}_{i}{f.suffix.lower()}'
                    if target.exists(): target.unlink()
                    if a.copy: shutil.copy2(f,target)
                    else:
                        try: target.hardlink_to(f)
                        except OSError: shutil.copy2(f,target)
                counts[f'{task}/{split}/{name}']=len(items)
    print('Prepared dataset:')
    for k,v in counts.items(): print(f'  {k}: {v} images')
    print(f'Output: {out.resolve()}')
if __name__=='__main__': main()
