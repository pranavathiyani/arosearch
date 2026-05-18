---
title: AROsearch
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# AROsearch

Natural-language search over CARD's Antibiotic Resistance Ontology (ARO).

CARD's built-in search is keyword-only — typing "last-resort beta-lactamases"
returns nothing because those words aren't in the ARO descriptions. AROsearch
fixes that with hybrid retrieval: BM25 for exact gene name and acronym matches,
dense embeddings for natural-language understanding, fused with a query-adaptive
weighting so each query gets the right balance automatically.

## Try it

**Live demo:** https://huggingface.co/spaces/pranavathiyani/arosearch

## How to build / deploy

The workflow has three steps. You only build once unless CARD releases a new
version of the ARO; after that the app just serves the artifacts.

### 1. Build indices on Colab (one-time, ~5 minutes)

Open `build_index_colab.ipynb` in Google Colab, switch the runtime to T4 GPU,
and run all cells. The notebook downloads CARD data, builds both indices, runs
smoke-test queries, and packages the three artifacts into a zip you download
to your machine.

### 2. Commit artifacts to this repo

Unzip the artifacts into `data/`:

```
data/
├── aro_index.faiss      (~26 MB — FAISS dense index)
├── aro_bm25.pkl         (~320 KB — BM25S sparse index)
└── aro_meta.parquet     (~267 KB — ARO metadata)
```

Then `git add data/ && git commit && git push`.

### 3. Deploy to HuggingFace Spaces

Create a new Gradio Space, point it at this GitHub repo, and Spaces will
auto-build using `requirements.txt` and run `app.py`. The free CPU tier is
sufficient for query-time inference.

## How it works

For each ARO entry we build two text representations:

- **Lexical view**: name + gene family. Short, term-dense — good for BM25.
- **Semantic view**: name + description + drug classes + mechanisms. Prose-shaped — good for dense embeddings.

At query time:

1. Encode the query with Qwen3-Embedding-0.6B, retrieve top-50 by cosine over the semantic index.
2. Tokenize the query, retrieve top-50 by BM25 over the lexical index.
3. Min-max normalize BM25 scores to `[0, 1]`.
4. Fuse with sigmoid-adaptive α:

   ```
   α = 1 / (1 + exp(-10 · (BM25_norm − 0.5)))
   score = α · BM25_norm + (1 − α) · cosine
   ```

   Strong lexical match → α → 1, BM25 dominates. Weak lexical match → α → 0, embeddings dominate.

5. Return top-K ranked results with all four signals visible so you can audit why a result ranked where it did.

The fusion follows Wijesekara et al. 2026 (VectorSage, *Bioinformatics Advances*).

## Local development

To run the app locally pointing at committed artifacts:

```bash
git clone https://github.com/pranavathiyani/arosearch
cd arosearch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py        # serves Gradio UI on http://localhost:7860
```

## License & citation

Code: MIT.

ARO data is licensed under CC-BY 4.0 by McMaster University and accessed via
[github.com/arpcard/aro](https://github.com/arpcard/aro). If you use AROsearch
in your work, please cite CARD:

> Alcock BP, et al. CARD 2023: expanded curation, support for machine learning,
> and resistome prediction at the Comprehensive Antibiotic Resistance Database.
> *Nucleic Acids Research* 51, D690–D699 (2023).

## Author

Pranavathiyani Gnanasekar · SASTRA Deemed University · pranavathiyani@scbt.sastra.edu
