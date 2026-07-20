import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED           = 42
NUM_LAYERS     = 12
EMBEDDINGS_DIR = "data/embeddings"
RESULTS_DIR    = "results"

np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

with open("data/crops.json") as f:
    records = json.load(f)

domains       = sorted(set(r["domain"] for r in records))
domain_to_idx = {d: i for i, d in enumerate(domains)}
idx_to_domain = {v: k for k, v in domain_to_idx.items()}
domain_names  = [idx_to_domain[i] for i in range(len(domains))]

train_layout = np.load(os.path.join(EMBEDDINGS_DIR, "train_layout_labels.npy"))
held_layout  = np.load(os.path.join(EMBEDDINGS_DIR, "held_layout_labels.npy"))

train_domain = np.array([domain_to_idx[r["domain"]] for r in records if r["split"] == "train"])
held_domain  = np.array([domain_to_idx[r["domain"]] for r in records if r["split"] == "held_out"])
all_domain   = np.concatenate([train_domain, held_domain], axis=0)

layout_acc_seen  = []
layout_acc_held  = []
layout_f1_seen   = []
layout_f1_held   = []
domain_acc_cv    = []
domain_f1_cv     = []

layout_names = [
    "caption", "footnote", "formula", "list_item", "page_footer",
    "page_header", "picture", "section_header", "table", "text", "title",
]

for layer in range(NUM_LAYERS):
    print("layer", layer)

    train_emb = np.load(os.path.join(EMBEDDINGS_DIR, "layer_" + str(layer).zfill(2) + "_train.npy")).astype(np.float32)
    held_emb  = np.load(os.path.join(EMBEDDINGS_DIR, "layer_" + str(layer).zfill(2) + "_held.npy")).astype(np.float32)
    all_emb   = np.concatenate([train_emb, held_emb], axis=0)

    scaler           = StandardScaler()
    train_emb_sc     = scaler.fit_transform(train_emb)
    held_emb_sc      = scaler.transform(held_emb)
    all_emb_sc       = scaler.transform(all_emb)

    probe_a = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", random_state=SEED, n_jobs=-1)
    probe_a.fit(train_emb_sc, train_layout)

    preds_seen = probe_a.predict(train_emb_sc)
    preds_held = probe_a.predict(held_emb_sc)

    layout_acc_seen.append(accuracy_score(train_layout, preds_seen))
    layout_acc_held.append(accuracy_score(held_layout,  preds_held))
    layout_f1_seen.append(f1_score(train_layout, preds_seen, average="macro", zero_division=0))
    layout_f1_held.append(f1_score(held_layout,  preds_held, average="macro", zero_division=0))

    X_tr, X_te, y_tr, y_te = train_test_split(
        all_emb_sc, all_domain, test_size=0.2, random_state=SEED, stratify=all_domain
    )
    probe_b = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", random_state=SEED, n_jobs=-1)
    probe_b.fit(X_tr, y_tr)
    preds_b = probe_b.predict(X_te)

    domain_acc_cv.append(accuracy_score(y_te, preds_b))
    domain_f1_cv.append(f1_score(y_te, preds_b, average="macro", zero_division=0))

    print("  probe_a seen", round(layout_acc_seen[-1], 4), "held", round(layout_acc_held[-1], 4))
    print("  probe_b domain", round(domain_acc_cv[-1], 4))

layers = list(range(1, NUM_LAYERS + 1))
chance = 1.0 / len(domains)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(layers, layout_acc_seen, marker="o", label="seen domains")
ax1.plot(layers, layout_acc_held, marker="o", label="unseen domains")
ax1.axhline(y=chance, color="gray", linestyle="--", label="chance")
ax1.set_xlabel("Layer")
ax1.set_ylabel("Accuracy")
ax1.set_title("Probe A: Layout Class Accuracy per Layer")
ax1.set_xticks(layers)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(layers, domain_acc_cv, marker="s", color="darkorange", label="domain probe")
ax2.axhline(y=chance, color="gray", linestyle="--", label="chance")
ax2.set_xlabel("Layer")
ax2.set_ylabel("Accuracy")
ax2.set_title("Probe B: Domain Identity Accuracy per Layer")
ax2.set_xticks(layers)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "layerwise_probe.png"), dpi=150)
plt.close()
print("plot saved to results/layerwise_probe.png")

results = {
    "layout_acc_seen" : layout_acc_seen,
    "layout_acc_held" : layout_acc_held,
    "layout_f1_seen"  : layout_f1_seen,
    "layout_f1_held"  : layout_f1_held,
    "domain_acc"      : domain_acc_cv,
    "domain_f1"       : domain_f1_cv,
    "chance"          : chance,
}

import json
with open(os.path.join(RESULTS_DIR, "layerwise_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nsummary (layer 1 → 12):")
print("layer  layout_seen  layout_held   domain_acc")
for i in range(NUM_LAYERS):
    print(
        str(i + 1).rjust(5),
        str(round(layout_acc_seen[i], 4)).rjust(12),
        str(round(layout_acc_held[i], 4)).rjust(12),
        str(round(domain_acc_cv[i], 4)).rjust(12),
    )