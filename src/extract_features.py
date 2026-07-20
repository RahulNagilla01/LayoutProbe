
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights

sys.path.insert(0, os.path.dirname(__file__))
from dataset import LayoutProbeDataset

SEED           = 42
BATCH_SIZE     = 128
NUM_LAYERS     = 12
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH     = "results/best_vit.pth"
EMBEDDINGS_DIR = "data/embeddings"


def build_model(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Linear(768, num_classes)
    return model


def extract_all_layers(model, loader):
    model.eval()

    layer_outputs   = {}
    layer_collected = {i: [] for i in range(NUM_LAYERS)}
    all_layout      = []
    all_domain      = []

    def make_hook(idx):
        def hook(module, input, output):
            cls        = output[:, 0]
            patch_mean = output[:, 1:].mean(dim=1)
            combined   = torch.cat([cls, patch_mean], dim=1)
            layer_outputs[idx] = combined.detach().cpu().numpy().astype(np.float16)
        return hook

    hooks = []
    for i in range(NUM_LAYERS):
        h = model.encoder.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for batch_idx, (imgs, layout_labels, domain_labels) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            _    = model(imgs)

            for i in range(NUM_LAYERS):
                layer_collected[i].append(layer_outputs[i])

            all_layout.append(layout_labels.numpy())
            all_domain.append(domain_labels.numpy())

            if batch_idx % 50 == 0:
                done  = batch_idx * BATCH_SIZE
                total = len(loader.dataset)
                print(done, "/", total)

    for h in hooks:
        h.remove()

    layer_embeddings = {i: np.concatenate(layer_collected[i], axis=0) for i in range(NUM_LAYERS)}
    layout_out       = np.concatenate(all_layout, axis=0)
    domain_out       = np.concatenate(all_domain, axis=0)

    return layer_embeddings, layout_out, domain_out


def run():
    torch.manual_seed(SEED)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    train_ds    = LayoutProbeDataset(split="train")
    held_out_ds = LayoutProbeDataset(split="held_out")
    num_classes = train_ds.num_layout_classes()

    model = build_model(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    for param in model.parameters():
        param.requires_grad = False

    print("device       :", DEVICE)
    print("train crops  :", len(train_ds))
    print("held crops   :", len(held_out_ds))
    print("layers       :", NUM_LAYERS)
    print("emb dim      : 1536")

    train_loader = DataLoader(train_ds,    batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    held_loader  = DataLoader(held_out_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\ntrain extraction")
    train_emb, train_layout, train_domain = extract_all_layers(model, train_loader)

    for i in range(NUM_LAYERS):
        np.save(os.path.join(EMBEDDINGS_DIR, "layer_" + str(i).zfill(2) + "_train.npy"), train_emb[i])
    np.save(os.path.join(EMBEDDINGS_DIR, "train_layout_labels.npy"), train_layout)
    np.save(os.path.join(EMBEDDINGS_DIR, "train_domain_labels.npy"), train_domain)

    print("\nheld-out extraction")
    held_emb, held_layout, held_domain = extract_all_layers(model, held_loader)

    for i in range(NUM_LAYERS):
        np.save(os.path.join(EMBEDDINGS_DIR, "layer_" + str(i).zfill(2) + "_held.npy"), held_emb[i])
    np.save(os.path.join(EMBEDDINGS_DIR, "held_layout_labels.npy"), held_layout)
    np.save(os.path.join(EMBEDDINGS_DIR, "held_domain_labels.npy"), held_domain)

    print("\ndone")
    for i in range(NUM_LAYERS):
        print("layer", i, "train:", train_emb[i].shape, "held:", held_emb[i].shape)


if __name__ == "__main__":
    run()
