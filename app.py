"""
AROsearch — Gradio web interface for HF Spaces.

Loads the searcher once at startup, serves a single-text-box UI.
For each result, shows ARO ID, name, fused score, and the underlying
BM25/cosine/alpha values so users can see why a result ranked where it did.
"""

from __future__ import annotations

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

# Load once at startup (HF Spaces will keep this in memory).
searcher = AROSearcher()


def run_search(query: str, top_k: int) -> pd.DataFrame:
    hits = searcher.search(query, top_k=int(top_k))
    if not hits:
        return pd.DataFrame(columns=["ARO ID", "Name", "Score", "α", "BM25", "Cosine", "Drug Classes", "Mechanisms"])

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
        })
    return pd.DataFrame(rows)


with gr.Blocks(title="AROsearch", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AROsearch
        Natural-language search over CARD's Antibiotic Resistance Ontology.

        Hybrid retrieval: BM25 (lexical) + Qwen3-Embedding-0.6B (semantic),
        fused with a sigmoid-adaptive weighting that shifts between the two
        based on how well the query lexically matches the corpus.

        **α** is the lexical weight — high α means the result ranked mostly
        on keyword match; low α means it ranked on semantic similarity.

        Data: [ARO](https://github.com/arpcard/aro) (CC-BY 4.0) ·
        Cite CARD: [Alcock et al. 2023](https://doi.org/10.1093/nar/gkac920)
        """
    )

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
        max_height=600,
    )

    submit_btn.click(fn=run_search, inputs=[query_box, top_k_slider], outputs=results_table)
    query_box.submit(fn=run_search, inputs=[query_box, top_k_slider], outputs=results_table)


if __name__ == "__main__":
    demo.launch()
