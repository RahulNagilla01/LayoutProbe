import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

VAL_PARQUET  = "/teamspace/studios/this_studio/Val.parquet"
TEST_PARQUET = "/teamspace/studios/this_studio/Test.parquet"
OUTPUT_DIR   = "data/crops"
LABELS_FILE  = "data/crops.json"

MIN_CROP_SIZE = 10

CATEGORY_MAP = {
    1:  "caption",
    2:  "footnote",
    3:  "formula",
    4:  "list_item",
    5:  "page_footer",
    6:  "page_header",
    7:  "picture",
    8:  "section_header",
    9:  "table",
    10: "text",
    11: "title",
}

DOMAIN_MAP = {
    "financial_reports"   : 0,
    "scientific_articles" : 1,
    "laws_and_regulations": 2,
    "government_tenders"  : 3,
    "manuals"             : 4,
    "patents"             : 5,
}

TRAIN_DOMAINS = {"manuals", "laws_and_regulations"}
HELD_OUT_DOMAINS = {"patents", "government_tenders", "financial_reports"}


def extract_crops(df, split_name):
    records = []
    crop_idx = 0

    for row_idx, row in df.iterrows():
        meta        = row["metadata"]
        doc_category = meta.get("doc_category", None)

        if doc_category not in DOMAIN_MAP:
            continue

        try:
            img = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
        except Exception:
            continue

        img_w, img_h = img.size
        bboxes       = row["bboxes"]
        category_ids = row["category_id"]

        for bbox, cat_id in zip(bboxes, category_ids):
            if cat_id is None or np.isnan(cat_id):
                continue

            cat_id = int(cat_id)
            if cat_id not in CATEGORY_MAP:
                continue

            x, y, w, h = bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))

            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                continue

            crop = img.crop((x1, y1, x2, y2))
            fname = f"{split_name}_{crop_idx:06d}.jpg"
            crop.save(os.path.join(OUTPUT_DIR, fname), quality=90)

            records.append({
                "filename"    : fname,
                "layout_class": cat_id,
                "layout_name" : CATEGORY_MAP[cat_id],
                "domain"      : doc_category,
                "domain_id"   : DOMAIN_MAP[doc_category],
                "split"       : "train" if doc_category in TRAIN_DOMAINS else "held_out",
            })

            crop_idx += 1

        if row_idx % 100 == 0:
            print(f"  [{split_name}] processed {row_idx} pages, {crop_idx} crops so far")

    return records


def build():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading parquet files...")
    val_df  = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    print(f"Val pages : {len(val_df)}")
    print(f"Test pages: {len(test_df)}")

    print("\nExtracting val crops...")
    val_records  = extract_crops(val_df,  "val")

    print("\nExtracting test crops...")
    test_records = extract_crops(test_df, "test")

    all_records = val_records + test_records

    with open(LABELS_FILE, "w") as f:
        json.dump(all_records, f)

    print(f"\nTotal crops saved : {len(all_records)}")
    print(f"Crops directory   : {OUTPUT_DIR}")
    print(f"Labels file       : {LABELS_FILE}")

    train_crops    = [r for r in all_records if r["split"] == "train"]
    held_out_crops = [r for r in all_records if r["split"] == "held_out"]
    print(f"\nTrain domains  (manuals + laws)          : {len(train_crops)} crops")
    print(f"Held-out domains (patents + tenders + fin): {len(held_out_crops)} crops")

    print("\nLayout class distribution:")
    from collections import Counter
    class_counts = Counter(r["layout_name"] for r in all_records)
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s}: {cnt}")

    print("\nDomain distribution:")
    domain_counts = Counter(r["domain"] for r in all_records)
    for dom, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {dom:25s}: {cnt}")


if __name__ == "__main__":
    build()