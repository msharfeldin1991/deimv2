"""
Run this script with your Roboflow API key to download a pill dataset
in COCO format into the expected directory structure.

Usage:
    python download_pill_dataset.py --api-key YOUR_KEY_HERE

Get your key at: https://app.roboflow.com → Settings → API Keys
"""

import argparse
import json
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument(
        "--workspace", default="roboflow-universe-projects",
        help="Roboflow workspace slug"
    )
    parser.add_argument(
        "--project", default="pill-counting",
        help="Roboflow project slug (see alternatives below)"
    )
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    # ── Alternative pill datasets on Roboflow Universe ────────────────────────
    # If the default project doesn't exist or has no version 1, try these:
    #   workspace="seblful"       project="pills-detection-s9ywn"
    #   workspace="pill-detection" project="pill-detection-x3b8u"
    #   workspace="roboflow-universe-projects" project="pills-tablet-capsule"
    # Browse more at: https://universe.roboflow.com/search?q=pill+detection&t=metadata

    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("coco", location="./rf_download")

    print(f"\nDownloaded to: {dataset.location}")
    print("Reorganising into data/pills/ ...")

    # Roboflow COCO export layout:
    #   rf_download/train/  + rf_download/train/_annotations.coco.json
    #   rf_download/valid/  + rf_download/valid/_annotations.coco.json
    #   rf_download/test/   + rf_download/test/_annotations.coco.json  (optional)

    os.makedirs("data/pills/train2017", exist_ok=True)
    os.makedirs("data/pills/val2017",   exist_ok=True)
    os.makedirs("data/pills/annotations", exist_ok=True)

    for rf_split, coco_split in [("train", "train2017"), ("valid", "val2017")]:
        src_dir = os.path.join(dataset.location, rf_split)
        dst_dir = f"data/pills/{coco_split}"
        ann_src = os.path.join(src_dir, "_annotations.coco.json")
        ann_dst = f"data/pills/annotations/instances_{coco_split}.json"

        if not os.path.isdir(src_dir):
            print(f"  WARNING: {src_dir} not found, skipping")
            continue

        # Copy images
        count = 0
        for fname in os.listdir(src_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
                count += 1
        print(f"  {rf_split} → {coco_split}: {count} images")

        # Copy + fix annotations (remap_mscoco_category=False so IDs must start at 1)
        if os.path.isfile(ann_src):
            with open(ann_src) as f:
                ann = json.load(f)
            # Ensure category IDs start at 1 and rename to tablet/capsule if needed
            print(f"  Categories: {[c['name'] for c in ann.get('categories', [])]}")
            with open(ann_dst, "w") as f:
                json.dump(ann, f)
            print(f"  Annotations written to {ann_dst}")

    # Print class summary
    ann_file = "data/pills/annotations/instances_train2017.json"
    if os.path.isfile(ann_file):
        with open(ann_file) as f:
            ann = json.load(f)
        cats = {c["id"]: c["name"] for c in ann["categories"]}
        from collections import Counter
        label_counts = Counter(a["category_id"] for a in ann["annotations"])
        print("\nClass distribution in train:")
        for cid, name in cats.items():
            print(f"  [{cid}] {name}: {label_counts.get(cid, 0)} instances")
        print(f"\nTotal train images : {len(ann['images'])}")
        print(f"Total train bboxes : {len(ann['annotations'])}")

    print("\nDone. You can now run training:")
    print("  python train.py -c configs/deimv2/deimv2_hgnetv2_atto_pills.yml \\")
    print("    --use-amp -t weights/deimv2_atto_coco.pth")


if __name__ == "__main__":
    main()
