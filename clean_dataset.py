import cv2
from pathlib import Path
import os

def clean_directory(split):
    images_dir = Path(f"datasets/merged/{split}/images")
    labels_dir = Path(f"datasets/merged/{split}/labels")
    
    if not images_dir.exists():
        return
        
    removed = 0
    for img_path in images_dir.iterdir():
        if img_path.is_file():
            im = cv2.imread(str(img_path))
            if im is None:
                print(f"Removing corrupt image: {img_path}")
                img_path.unlink()
                
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    label_path.unlink()
                removed += 1
                
    if removed > 0:
        # Also remove the cache file so YOLO regenerates it
        cache_path = Path(f"datasets/merged/{split}/labels.cache")
        if cache_path.exists():
            cache_path.unlink()
            print(f"Removed cache: {cache_path}")
            
    print(f"{split}: removed {removed} corrupt images")

for split in ["train", "val", "test"]:
    clean_directory(split)
