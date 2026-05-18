"""
AROsearch — Gradio web interface for HF Spaces.

Loads the searcher once at startup, serves a single-text-box UI.
For each result, shows ARO ID (linked to CARD), name, fused score, and the
underlying BM25/cosine/alpha values so users can see why a result ranked where
it did. Results can be downloaded as CSV.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from search import AROSearcher

EXAMPLES = [
    "enzymes that hydrolyze last-resort beta-lactams",
    "carbapenem resistance in Klebsiella pneumoniae",
    "efflux pump fluoroquinolone resistance",
    "KPC-2",
    "ribosomal protection protein tetracycline",
    "vancomycin resistance enterococci",
    "colistin resistance plasmid-mediated",
    "16S rRNA methyltransferase aminoglycoside",
]

INTRO_MARKDOWN = """
# 🧬 AROsearch

Natural-language search over CARD's Antibiotic Resistance Ontology.

Type a gene name (KPC-2), a phrase (carbapenem resistance Klebsiella), or a full
question (enzymes that hydrolyze last-resort beta-lactams) — AROsearch fuses
keyword and semantic retrieval to find the right ARO terms either way.
"""

HOW_IT_WORKS_MARKDOWN = """
### Two indices, one fused score

For each of the ~6,400 ARO entries, AROsearch builds two text representations:

- **Lexical view** — name + gene family. Term-dense; BM25 ranks it well.
- **Semantic view** — name + description + drug classes + mechanisms.
  Prose-shaped; dense embeddings handle it well.

At query time the system runs both retrievers in parallel and fuses the
results with a sigmoid-adaptive weighting:

```
α = 1 / (1 + exp(-10 · (BM25_norm − 0.5)))
score = α · BM25_norm + (1 − α) · cosine
```

When the query has a strong keyword match (e.g. an exact gene name),
α → 1 and BM25 dominates. When it's a natural-language phrase with no
direct lexical overlap, α → 0 and the embedding similarity drives the
ranking. This adaptive fusion is adapted from Wijesekara et al. 2026
(VectorSage, *Bioinformatics Advances*).

### Stack

- **Embeddings:** Qwen3-Embedding-0.6B (1024-d, fp16 on CPU)
- **Vector index:** FAISS (IndexFlatIP, exact search — corpus is small)
- **Sparse index:** BM25S (fast pure-Python BM25)
- **UI:** Gradio · **Hosted on:** HuggingFace Spaces (free CPU tier)

### How to read the result columns

- **Score** — fused final score in `[0, 1]`. Higher is better.
- **α** — lexical weight at fusion time. High α means the hit ranked mostly
  on keyword match; low α means on semantic similarity.
- **BM25** — min-max normalized BM25 score (lexical signal).
- **Cosine** — cosine similarity between query and document embeddings.
- **CARD link** — opens the full ARO record at card.mcmaster.ca.

You can use α to audit *why* something ranked where it did.

### Out-of-domain queries

If you search something that's not in the AMR vocabulary (e.g. "pikachu",
"chocolate cake"), expect low overall scores across the board. The system
will surface its closest embedding neighbors but they won't be meaningful.
Low BM25 + low cosine = the system is telling you it doesn't know.
"""

DISCLAIMER_MARKDOWN = """
### ⚠️ Disclaimer

AROsearch is a **research tool**, not a clinical decision support system.
Results reflect what CARD's ARO contains and how the retrieval model interprets
your query — they are not medical advice and must not be used for diagnosis,
treatment, or antimicrobial prescribing decisions. Always verify any finding
against primary literature and CARD itself before acting on it.

The ARO data is a snapshot of the latest CARD release at the time of indexing
and may lag behind the live database. Re-run the indexing pipeline to refresh.

### License & data

- **Code:** MIT
- **Data:** Antibiotic Resistance Ontology (ARO), © McMaster University,
  CC-BY 4.0, available at [github.com/arpcard/aro](https://github.com/arpcard/aro)
- **Please cite CARD** if you use AROsearch in published work:
  Alcock BP, et al. CARD 2023: expanded curation, support for machine learning,
  and resistome prediction at the Comprehensive Antibiotic Resistance Database.
  *Nucleic Acids Research* 51, D690–D699 (2023).
  [doi:10.1093/nar/gkac920](https://doi.org/10.1093/nar/gkac920)

### Built by

[**Pranavathiyani Gnanasekar**](https://pranavathiyani.github.io) —
Division of Bioinformatics, SASTRA Deemed University.
Built with assistance from [Claude](https://claude.ai) (Anthropic).

[Source code on GitHub](https://github.com/pranavathiyani/arosearch) ·
Feedback / issues welcome.
"""

# Load once at startup
searcher = AROSearcher()


def _card_url(aro_id: str) -> str:
    """ARO:3002312 -> https://card.mcmaster.ca/aro/3002312"""
    numeric = aro_id.replace("ARO:", "")
    return f"https://card.mcmaster.ca/aro/{numeric}"


def run_search(query: str, top_k: int):
    """Returns (results dataframe, downloadable CSV file path or None)."""
    hits = searcher.search(query, top_k=int(top_k))
    if not hits:
        empty = pd.DataFrame(
            columns=["ARO ID", "Name", "Score", "α", "BM25", "Cosine",
                     "Drug Classes", "Mechanisms", "CARD link"]
        )
        return empty, None

    rows = []
    for h in hits:
        rows.append({
            "ARO ID": h.aro_id,
            "Name": h.name,
            "Score": round(h.score, 3),
            "α": round(h.alpha, 2),
            "BM25": round(h.bm25_norm, 2),
            "Cosine": round(h.cosim, 2),
            "Drug Classes": h.drug_classes or "—",
            "Mechanisms": h.mechanisms or "—",
            "CARD link": _card_url(h.aro_id),
        })
    df = pd.DataFrame(rows)

    # Write the full result (including the query and full description) to a temp CSV
    # for download. We include description here even though it's not shown in the table.
    download_rows = []
    for h in hits:
        download_rows.append({
            "query": query,
            "aro_id": h.aro_id,
            "name": h.name,
            "description": h.description,
            "drug_classes": h.drug_classes,
            "mechanisms": h.mechanisms,
            "families": h.families,
            "score": round(h.score, 4),
            "alpha": round(h.alpha, 4),
            "bm25_norm": round(h.bm25_norm, 4),
            "cosine": round(h.cosim, 4),
            "card_url": _card_url(h.aro_id),
        })
    download_df = pd.DataFrame(download_rows)

    safe_query = "".join(c if c.isalnum() else "_" for c in query[:30])
    tmp_path = Path(tempfile.gettempdir()) / f"arosearch_{safe_query or 'results'}.csv"
    download_df.to_csv(tmp_path, index=False)

    return df, str(tmp_path)


with gr.Blocks(title="AROsearch", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO_MARKDOWN)

    with gr.Row():
        query_box = gr.Textbox(
            label="Query",
            placeholder="Type a gene name, mechanism, or natural-language description...",
            scale=4,
        )
        top_k_slider = gr.Slider(
            minimum=5, maximum=25, value=10, step=1,
            label="Results", scale=1,
        )

    submit_btn = gr.Button("Search", variant="primary")

    gr.Examples(examples=EXAMPLES, inputs=query_box, label="Try these")

    results_table = gr.Dataframe(
        label="Results",
        wrap=True,
        datatype=["str", "str", "number", "number", "number", "number",
                  "str", "str", "markdown"],
    )

    download_file = gr.File(label="Download results (CSV)", visible=True)

    with gr.Accordion("How it works", open=False):
        gr.Markdown(HOW_IT_WORKS_MARKDOWN)

    with gr.Accordion("Disclaimer, license & credits", open=False):
        gr.Markdown(DISCLAIMER_MARKDOWN)

    submit_btn.click(
        fn=run_search,
        inputs=[query_box, top_k_slider],
        outputs=[results_table, download_file],
    )
    query_box.submit(
        fn=run_search,
        inputs=[query_box, top_k_slider],
        outputs=[results_table, download_file],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
