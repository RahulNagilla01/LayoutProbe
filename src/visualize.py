import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

sys.path.insert(0, os.path.dirname(__file__))

SEED           = 42
LAYER          = 11
SUBSET         = 4000
EMBEDDINGS_DIR = "data/embeddings"
RESULTS_DIR    = "results"

np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

train_emb    = np.load(os.path.join(EMBEDDINGS_DIR, "layer_" + str(LAYER).zfill(2) + "_train.npy")).astype(np.float32)
held_emb     = np.load(os.path.join(EMBEDDINGS_DIR, "layer_" + str(LAYER).zfill(2) + "_held.npy")).astype(np.float32)
train_layout = np.load(os.path.join(EMBEDDINGS_DIR, "train_layout_labels.npy"))
held_layout  = np.load(os.path.join(EMBEDDINGS_DIR, "held_layout_labels.npy"))

with open("data/crops.json") as f:
    records = json.load(f)

domains       = sorted(set(r["domain"] for r in records))
domain_to_idx = {d: i for i, d in enumerate(domains)}
idx_to_domain = {v: k for k, v in domain_to_idx.items()}

train_domain = np.array([domain_to_idx[r["domain"]] for r in records if r["split"] == "train"])
held_domain  = np.array([domain_to_idx[r["domain"]] for r in records if r["split"] == "held_out"])

all_emb    = np.concatenate([train_emb, held_emb], axis=0)
all_layout = np.concatenate([train_layout, held_layout], axis=0)
all_domain = np.concatenate([train_domain, held_domain], axis=0)

layout_names = [
    "caption", "footnote", "formula", "list_item", "page_footer",
    "page_header", "picture", "section_header", "table", "text", "title",
]
domain_names = [idx_to_domain[i] for i in range(len(domains))]
table_idx    = layout_names.index("table")

scaler     = StandardScaler()
all_emb_sc = scaler.fit_transform(all_emb)

idx           = np.random.choice(len(all_emb_sc), size=SUBSET, replace=False)
subset_emb    = all_emb_sc[idx]
subset_layout = all_layout[idx]
subset_domain = all_domain[idx]

print("running PCA to 50 dims...")
pca_50   = PCA(n_components=50, random_state=SEED)
emb_50   = pca_50.fit_transform(subset_emb)
print("variance explained (50 components):", round(pca_50.explained_variance_ratio_.sum(), 4))

print("running t-SNE...")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=40, max_iter=1000)
emb_tsne = tsne.fit_transform(emb_50)

print("running UMAP...")
reducer  = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
emb_umap = reducer.fit_transform(emb_50)

print("running PCA 2D...")
pca_2d  = PCA(n_components=2, random_state=SEED)
emb_pca = pca_2d.fit_transform(subset_emb)
print("variance explained (2 components):", round(pca_2d.explained_variance_ratio_.sum(), 4))


def plot_embeddings(emb_2d, labels, names, title, fname, cmap="tab20"):
    n_classes = len(names)
    colors    = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_classes))
    fig, ax   = plt.subplots(figsize=(10, 7))
    for i, name in enumerate(names):
        mask = labels == i
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[colors[i]], label=name.replace("_", " "),
                   s=5, alpha=0.6)
    ax.set_title(title)
    ax.legend(fontsize=7, markerscale=3, loc="upper right")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close()
    print("saved", fname)


plot_embeddings(emb_tsne, subset_layout, layout_names,
                "t-SNE  Layer 12 by Layout Class", "tsne_layout.png")

plot_embeddings(emb_tsne, subset_domain, domain_names,
                "t-SNE  Layer 12 by Domain", "tsne_domain.png", cmap="tab10")

plot_embeddings(emb_umap, subset_layout, layout_names,
                "UMAP  Layer 12 by Layout Class", "umap_layout.png")

plot_embeddings(emb_umap, subset_domain, domain_names,
                "UMAP  Layer 12 by Domain", "umap_domain.png", cmap="tab10")

plot_embeddings(emb_pca, subset_layout, layout_names,
                "PCA  Layer 12 by Layout Class", "pca_layout.png")

plot_embeddings(emb_pca, subset_domain, domain_names,
                "PCA  Layer 12 by Domain", "pca_domain.png", cmap="tab10")

print("\ntable clustering analysis...")
table_mask   = all_layout == table_idx
table_emb    = all_emb_sc[table_mask]
table_domain = all_domain[table_mask]

print("table crops:", table_mask.sum())
for d_idx, d_name in enumerate(domain_names):
    print(" ", d_name, ":", (table_domain == d_idx).sum())

sub_idx      = np.random.choice(len(table_emb), size=min(1500, len(table_emb)), replace=False)
table_emb_s  = table_emb[sub_idx]
table_dom_s  = table_domain[sub_idx]

t_pca        = PCA(n_components=min(50, len(table_emb_s) - 1), random_state=SEED)
t_emb_50     = t_pca.fit_transform(table_emb_s)

t_umap       = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=20, min_dist=0.1)
t_emb_2d     = t_umap.fit_transform(t_emb_50)

colors       = plt.cm.tab10(np.linspace(0, 1, len(domain_names)))
fig, ax      = plt.subplots(figsize=(9, 6))
for d_idx, d_name in enumerate(domain_names):
    mask = table_dom_s == d_idx
    if mask.sum() > 0:
        ax.scatter(t_emb_2d[mask, 0], t_emb_2d[mask, 1],
                   c=[colors[d_idx]], label=d_name.replace("_", " "),
                   s=8, alpha=0.7)

ax.set_title("Table crops only — UMAP by domain\n(do tables from different domains cluster together?)")
ax.legend(fontsize=8, markerscale=2)
ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "table_clustering.png"), dpi=150)
plt.close()
print("saved table_clustering.png")

print("\nall done. check results/")