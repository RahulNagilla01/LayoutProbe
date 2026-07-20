# LayoutProbe

I was reading the IndicDLP paper from CVIT, IIIT Hyderabad and one question kept coming to my mind — when you train a Vision Transformer to understand document layouts, what exactly is it learning inside? Is it learning what a "table" looks like everywhere, or is it just memorizing what a table looks like in the specific documents it was trained on?

So I built this small experiment to find out.

---

## What is this project in simple words?

Okay so imagine you are teaching someone to identify furniture in photos. You show them photos from only two houses — a library and a law office. Now you ask them to identify furniture in a bank or a hospital. Will they do well?

Thats basically what I did here, but with a Vision Transformer (ViT) and document regions.

I trained the model on two types of documents. Then I froze the model completely (stopped all learning) and asked — **what information is stored inside the model right now?** Can I read out the layout class from it? Can I also read out which document type it came from? Are these two things mixed together or separate?

The answers were very interesting.

---

## The Research Question

> When a ViT is fine-tuned on document layout classification using only some document types, does it learn layout in a general way — or does it mix layout information with domain-specific patterns?

---

## Dataset

I used **DocLayNet** from IBM Research. Its a dataset of document images with bounding box annotations across 11 layout classes and 6 document types.

| Document Type | Role in Experiment | Number of Crops |
|--------------|-------------------|-----------------|
| Manuals | Training | 9,594 |
| Laws & Regulations | Training | 8,744 |
| Financial Reports | Held-out (never seen in training) | 22,805 |
| Scientific Articles | Held-out | 6,048 |
| Patents | Held-out | 6,375 |
| Government Tenders | Held-out | 4,037 |
| **Total** | | **57,603 crops** |

The 11 layout classes are: caption, footnote, formula, list item, page footer, page header, picture, section header, table, text, title.

---

## How I Did This (Step by Step)

**Step 1 — Extracted region crops**

For each document page, I used the bounding box annotations to cut out individual region images. So one document page becomes maybe 10-15 small crop images, each with a label like "this is a table" or "this is a title."

**Step 2 — Fine-tuned ViT-B/16**

ViT (Vision Transformer) works by dividing an image into 16×16 patches and applying self-attention across all patches. There is a special token called the CLS token which summarizes the entire image after passing through all 12 transformer layers.

I took a pretrained ViT-B/16 (pretrained on ImageNet) and fine-tuned it on layout classification using only the two training domains. After 10 epochs it reached **95% validation accuracy** on seen domains.

**Step 3 — Froze the model and extracted representations**

I completely stopped the model from learning (froze all weights). Then for every single crop — including the 4 unseen domains — I extracted:
- The **CLS token** (global summary, 768 dimensions)
- The **mean of all patch tokens** (spatial summary, 768 dimensions)
- Concatenated both → **1536 dimensional vector** per crop

I did this at all **12 transformer layers** separately. So for each crop I have 12 different representations showing how the information builds up layer by layer.

**Step 4 — Linear Probing**

This is the most important part. I trained two very simple logistic regression classifiers on these frozen representations:

- **Probe A** → predict the layout class (11 classes)
- **Probe B** → predict the document domain (6 classes)

Why logistic regression? Because its intentionally weak. If even this simple classifier can decode information from the frozen representations, it means that information was already clearly stored there. The probe didn't learn anything new — it just read what was already there.

---

## Results

### Fine-tuning Curve

The model trained well on seen domains:

![Training Curve](results/finetune_curve.png)

### Probe A — Layout Classification Across Layers

| Layer | Seen Domains | Unseen Domains |
|-------|-------------|----------------|
| 1 | 0.8164 | 0.5107 |
| 2 | 0.9093 | 0.5539 |
| 3 | 0.9426 | 0.5963 |
| 4 | 0.9663 | 0.6261 |
| 5 | 0.9862 | 0.6482 |
| 6 | 0.9962 | 0.6761 |
| 7 | 0.9994 | 0.6872 |
| 8 | 0.9998 | 0.6799 |
| 9 | 0.9998 | 0.6787 |
| 10 | 0.9998 | 0.6792 |
| 11 | 0.9998 | 0.6804 |
| 12 | 0.9998 | 0.6758 |

### Probe B — Domain Identity Across Layers

| Metric | Value |
|--------|-------|
| 5-fold CV Accuracy | 0.8668 ± 0.003 |
| Chance baseline | 0.17 |
| Signal above chance | +0.70 |

### Layerwise Probe Curves

![Layerwise Probe](results/layerwise_probe.png)

Look at this carefully. Domain accuracy (right plot) shoots up to 86% by layer 3 and stays there. Layout accuracy on unseen domains (left plot, orange line) keeps growing until layer 7 and then stops. This means the model first decides "which domain does this look like" in early layers, and then builds layout understanding on top of that domain context. By layer 5 both are fully baked in together.

### UMAP Visualizations

**By Layout Class:**

![UMAP Layout](results/umap_layout.png)

Each layout class forms its own region in the space. Tables (yellow), pictures (pink), page footers (purple) are all clearly separated. This shows the model did genuinely learn layout structure.

**By Domain:**

![UMAP Domain](results/umap_domain.png)

Now look at this plot using the same points as above. The large blob on the right (dim1 8-13) is almost all financial reports (blue). The top isolated clusters are laws and manuals. Domains are NOT randomly mixed — they have their own spatial regions.

Both signals sitting in the same space = entanglement.

**t-SNE by Layout Class:**

![t-SNE Layout](results/tsne_layout.png)

**t-SNE by Domain:**

![t-SNE Domain](results/tsne_domain.png)

### The Table Clustering Question

Do tables from all domains look the same to the model, or does it still see them as "financial report tables" vs "manual tables"?

![Table Clustering](results/table_clustering.png)

**This is the most honest result of the whole project.** Table is our best performing layout class (F1 = 0.77 on unseen domains). But look at this plot — financial report tables (blue) form their own separate big blob. Manual tables (pink) cluster at the top. Law tables (purple) are somewhere else.

Even the best class is still separated by domain underneath. The model never truly learned "what a table is" in a general sense. It learned "what a table looks like in these specific document types."

### Per-class F1 on Unseen Domains

| Class | F1 |
|-------|----|
| text | 0.7937 |
| table | 0.7730 |
| picture | 0.7081 |
| list item | 0.6326 |
| page footer | 0.6037 |
| section header | 0.4704 |
| page header | 0.1471 |
| title | 0.0300 |
| formula | 0.0232 |
| footnote | 0.0211 |
| caption | 0.0030 |

Visual classes (table, picture, text) transfer well because they look similar everywhere. Semantic classes (caption, title, footnote) completely fail because their appearance depends on the document's visual style.

---

## What Does All This Mean?

Three clear findings:

**1. The model learned two things at once without being told to.**
We only asked it to learn layout. But it also learned domain identity as a side effect. A simple linear classifier can predict document domain with 87% accuracy from the same representations — 70 points above chance. The information is clearly sitting there.

**2. Domain gets encoded before layout does.**
By layer 3 the domain signal is already strong. Layout on unseen domains keeps growing until layer 7. This tells us the model first processes "what kind of document is this" and then builds layout understanding on top of that context. So layout and domain are not just mixed — domain actually comes first.

**3. Even the best transferring class is domain-separated at representation level.**
Table has F1 of 0.77 on unseen domains, which seems decent. But the table clustering UMAP shows financial report tables and manual tables still live in separate neighborhoods. The generalization we see in accuracy numbers is not because the model learned truly domain-agnostic table representations — its because tables in different domains happen to share enough visual surface features (gridlines, cells) to still get classified correctly most of the time.

---

## Why Does This Matter?

For anyone building document AI systems across multiple languages and domains — like what IndicDLP is trying to do for Indian languages, or what BharatGen is building for multilingual foundation models — this is a real problem. If you train on some document types and expect the layout understanding to transfer cleanly to new ones, you will lose a lot on semantic layout classes specifically.

The fix probably requires either domain-adversarial training (explicitly removing domain signal from representations during fine-tuning) or training on a much broader set of document types from the beginning.

---

## Project Structure

```
LayoutProbe/
├── src/
│   ├── build_crops.py          extract region crops from DocLayNet
│   ├── dataset.py              PyTorch Dataset class with ViT transforms
│   ├── finetune.py             fine-tune ViT-B/16 on seen domains
│   ├── extract_features.py     extract CLS + patch mean at all 12 layers
│   ├── probe.py                layerwise Probe A + Probe B + curves
│   └── visualize.py            UMAP, t-SNE, PCA, table clustering
├── data/
│   ├── crops/                  57,603 extracted region images
│   ├── crops.json              metadata for all crops
│   └── embeddings/             layer-wise numpy arrays (12 layers × 2 splits)
├── results/
│   ├── best_vit.pth
│   ├── finetune_curve.png
│   ├── layerwise_probe.png
│   ├── umap_layout.png
│   ├── umap_domain.png
│   ├── tsne_layout.png
│   ├── tsne_domain.png
│   ├── pca_layout.png
│   ├── pca_domain.png
│   ├── table_clustering.png
│   └── layerwise_results.json
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/rahulnagilla/LayoutProbe
cd LayoutProbe
pip install torch torchvision scikit-learn pandas pyarrow pillow matplotlib umap-learn

# download val-00000-of-00003.parquet and test-00000-of-00003.parquet
# from ds4sd/DocLayNet-v1.1 on HuggingFace and place in project root

python src/build_crops.py
python src/finetune.py
python src/extract_features.py
python src/probe.py
python src/visualize.py
```

---

## References

- Pfitzmann et al. DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation. KDD 2022.
- Nath et al. IndicDLP: A Foundational Dataset for Multi-lingual and Multi-domain Document Layout Parsing. ICDAR 2025.
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
- Kumar et al. TexTAR: Textual Attribute Recognition in Multi-domain and Multi-lingual Document Images. ICDAR 2025.

---

*I am a second year B.Tech CSE (AI/ML) student at Bennett University. This project was motivated by reading Prof. Ravi Kiran Sarvadevabhatla's work on document analysis at CVIT, IIIT Hyderabad. Built this to understand representation learning in document understanding from first principles.*