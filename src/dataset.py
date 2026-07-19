import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_DIR  = "data/crops"
LABELS_FILE = "data/crops.json"

LAYOUT_CLASSES = {
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

DOMAIN_CLASSES = {
    0: "financial_reports",
    1: "scientific_articles",
    2: "laws_and_regulations",
    3: "government_tenders",
    4: "manuals",
    5: "patents",
}

VIT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])


class LayoutProbeDataset(Dataset):

    def __init__(self, json_file=LABELS_FILE, image_dir=IMAGE_DIR, split=None, transform=None):
        with open(json_file) as f:
            all_records = json.load(f)

        if split is not None:
            all_records = [r for r in all_records if r["split"] == split]

        self.records   = all_records
        self.image_dir = image_dir
        self.transform = transform or VIT_TRANSFORM

        layout_ids = sorted(set(r["layout_class"] for r in self.records))
        domain_ids = sorted(set(r["domain_id"]    for r in self.records))
        self.layout_id_to_idx = {lid: i for i, lid in enumerate(layout_ids)}
        self.domain_id_to_idx = {did: i for i, did in enumerate(domain_ids)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        img   = Image.open(os.path.join(self.image_dir, rec["filename"])).convert("RGB")
        img   = self.transform(img)
        layout = self.layout_id_to_idx[rec["layout_class"]]
        domain = self.domain_id_to_idx[rec["domain_id"]]
        return img, layout, domain

    def num_layout_classes(self):
        return len(self.layout_id_to_idx)

    def num_domain_classes(self):
        return len(self.domain_id_to_idx)

    def class_weights(self):
        from collections import Counter
        import torch
        counts = Counter(r["layout_class"] for r in self.records)
        total  = len(self.records)
        weights = []
        for lid in sorted(self.layout_id_to_idx.keys()):
            idx = self.layout_id_to_idx[lid]
            weights.append(total / (self.num_layout_classes() * counts[lid]))
        return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    torch.manual_seed(42)

    train_ds    = LayoutProbeDataset(split="train")
    held_out_ds = LayoutProbeDataset(split="held_out")
    full_ds     = LayoutProbeDataset(split=None)

    img, layout, domain = train_ds[0]

    print(f"Train crops    : {len(train_ds)}")
    print(f"Held-out crops : {len(held_out_ds)}")
    print(f"Total crops    : {len(full_ds)}")
    print(f"Layout classes : {train_ds.num_layout_classes()}")
    print(f"Domain classes : {train_ds.num_domain_classes()}")
    print(f"Image shape    : {img.shape}")
    print(f"Layout label   : {layout}")
    print(f"Domain label   : {domain}")
    print(f"Class weights  : {train_ds.class_weights()}")