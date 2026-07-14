# MegaRAG — Replication & Empirical Extensions

**SURAJ 2026 Summer Undergraduate Research Internship**
IIT Jodhpur, School of AI & Data Science (SAIDE) · Machine Intelligence Lab
Mentor: Dr. Divya Saxena · JRF Guidance: Aditya Sharma
Intern: Muskan (NIT Patna)

This repository is a replication of **MegaRAG** (Hsiao et al., ACL 2026, [arXiv:2512.20626](https://arxiv.org/abs/2512.20626)) — a multimodal knowledge-graph RAG framework — run end-to-end on a 788-page World History textbook using a fully local, zero-API-cost stack (Ollama + GME) on the IIT Jodhpur `dgx` SLURM cluster. It also contains two original empirical research extensions built on top of the replicated pipeline (see [Research Extensions](#-research-extensions-d1--d5)).

> For the unmodified upstream project, see [AI-Application-and-Integration-Lab/MegaRAG](https://github.com/AI-Application-and-Integration-Lab/MegaRAG).

---

## 🚀 Overview

MegaRAG builds a **Multimodal Knowledge Graph (MMKG)** from PDF documents — extracting entities and relationships from both OCR text and page images via a vision LLM — then answers questions by fusing graph traversal with multimodal page-image vector search.

**This replication's results (full corpus, 788 pages):**
| Metric | Value |
|---|---|
| Entities extracted | 3,657 |
| Relationships extracted | 1,910 |
| Page (chunk) vectors | 775 |
| Benchmark questions answered | 125 / 125, zero failures |

---

## 📦 Installation (Cluster Setup — IIT Jodhpur `dgx`)

### Requirements
- Python 3.10+
- GPU for embedding inference (A100 used here; GME requires CUDA)
- Local LLM via Ollama (no OpenAI key needed — see [Zero-Cost Local Setup](#-zero-cost-local-setup-ollama))

### ⚠️ Cluster gotcha: `anaconda3/2024` module corrupts `PATH`
On this cluster, loading the `anaconda3/2024` module breaks `PATH` resolution. **After every `conda activate`, re-export `PATH` explicitly**, or subsequent commands will silently resolve to the wrong interpreter:
```bash
conda activate megarag
export PATH="$CONDA_PREFIX/bin:$PATH"
```
Apply this after activating **either** environment (`mineru` or `megarag`).

### Step 1: Install MinerU
```bash
git clone -b release-1.3.6 https://github.com/opendatalab/MinerU.git
cd MinerU

conda create --name mineru python=3.10 -y
conda activate mineru
export PATH="$CONDA_PREFIX/bin:$PATH"
pip install -e .
pip install omegaconf   # required — not listed in MinerU's own dependencies

pip install huggingface_hub
wget https://raw.githubusercontent.com/opendatalab/MinerU/refs/heads/release-1.3.6/scripts/download_models_hf.py -O download_models_hf.py
sed -i "s|https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json|https://raw.githubusercontent.com/opendatalab/MinerU/release-1.3.6/magic-pdf.template.json|" download_models_hf.py
python download_models_hf.py
```

**Cluster-specific OCR fixes applied in this repo:**
- **PP-OCR weight remap:** the downloaded PP-OCRv5 weights were remapped to the PP-OCRv4 paths MinerU 1.3.6 expects (the release pins v4 paths internally).
- **Language flag removed:** the `-l en` flag was dropped from the MinerU invocation to align with the multilingual checkpoint actually being loaded — passing `-l en` against a multilingual checkpoint caused silent misbehavior.

### Step 2: Install MegaRAG
```bash
git clone https://github.com/AI-Application-and-Integration-Lab/MegaRAG.git
cd MegaRAG

conda activate mineru
export PATH="$CONDA_PREFIX/bin:$PATH"
pip install -r requirements_mineru.txt

conda create --name megarag python=3.10 -y
conda activate megarag
export PATH="$CONDA_PREFIX/bin:$PATH"
pip install -e .

cp .env.sh env.sh
# Fill in MINERU_PATH in env.sh. OPENAI_API_KEY is unused with Ollama — see below.

mkdir lib && cd lib
git clone --branch v1.4.3 https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .
```

### Step 3: Apply the refinement bug fix
```bash
conda activate megarag
python debug/apply_refinement_fix.py
```
Patches a known `IndexError` in `operate.py` triggered when any page produces zero entities (cover pages, blank pages, image-only pages). Safe to re-run.

---

## 🆓 Zero-Cost Local Setup (Ollama)

This replication runs entirely locally — no OpenAI spend.

- **LLM:** `qwen2.5:7b` (substitute for GPT-4o-mini) for entity extraction and querying
- **Embeddings:** `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` (GME) for multimodal text+image vectors

```bash
ollama pull qwen2.5:7b

export OPENAI_API_KEY="ollama"
export OPENAI_API_BASE="http://localhost:11434/v1"

ollama serve &
```

**⚠️ SLURM note:** `ollama serve` must be started **inside each SLURM compute job**, not on the login node. Login-node Ollama instances are not reachable from compute nodes and builds will hang waiting on the LLM endpoint.

**Set `example_number: 3` for local models:**
```yaml
# egs/<your_dataset>/conf/addon_params.yaml
example_number: 3
```
3-shot prompting is required for `qwen2.5:7b` to reliably follow the structured extraction format; with `example_number: 1` (the GPT-4o default) it frequently degrades to 0 Ent + 0 Rel per chunk.

---

## 🔍 Pre-Flight Validation (run before every build)

Run the debug suite on the login node before submitting any GPU job — takes ~5 minutes and avoids burning compute hours on a broken config.

```bash
conda activate megarag

# Static checks (no GPU, no Ollama needed)
python debug/apply_refinement_fix.py
python debug/E_refinement_audit.py
python debug/B_prompt_inspector.py

# Live check (Ollama must be running)
python debug/A_query_smoke_test.py

# Post-build validation
python debug/C_graph_inspector.py
```
Do not submit a build job until `A_query_smoke_test.py` shows all ✓ PASS.

---

## 📂 Corpus Used in This Replication

This repo's `egs/` recipe targets a 788-page World History textbook, validated in two staged passes:

1. **10-page pilot** — deliberate staged validation to confirm the pipeline, prompts, and OCR fixes were correct before committing GPU hours to the full corpus.
2. **788-page full-corpus run** — produced the final MMKG (3,657 entities, 1,910 relationships, 775 page vectors) used for all benchmark and research results.

```bash
cd egs/<your_dataset>
mkdir data          # place source PDF here
bash run_build_mmkg.sh
bash run_querying.sh
```

Watch build logs for `Chunk X of N extracted M Ent + K Rel` — both `M` and `K` should be `> 0` on every chunk. Persistent `0 Ent + 0 Rel` means the model isn't following the output format; raise `example_number` or check the OCR fixes above.

---

## 🏗️ Architecture

```
PDF → MinerU OCR → pages_content.json
                        │
                        ▼
              chunking_by_token_or_page()     1 chunk per page
                        │
                        ▼
              extract_entities()              vision LLM per page
              (+ refinement pass)             OCR text + page image + figures
                        │
                        ▼
              merge_nodes_and_edges()         builds knowledge graph
              GME multimodal embeddings       text + images in same vector space
                        │
                        ▼
              vdb_entities.json              entity vectors
              vdb_relationships.json         relationship vectors
              vdb_chunks.json                page image vectors (multimodal retrieval)
              graph_*.graphml                knowledge graph
```

**Query pipeline (`mix_two_step` mode):**
- **KG path:** keyword extraction → graph traversal → text chunk retrieval → LLM answer
- **Naive path:** GME vector search on page images → visual LLM answer
- **Merge:** both answers synthesized into the final response

---

## 🔬 Research Extensions (D1 & D5)

Beyond replication, this project contributes two falsifiable, empirically-tested research directions, motivated by structural gaps identified in the original paper.

### D1 — Minimal Sufficient Structure Hypothesis
The paper's A2 ablation conflates graph *structure* with graph *symbolic content*, leaving open which factor actually drives QA performance. This work isolates that question by building a **Pseudo-Entity Graph (PEG)** — a zero-LLM-cost graph structure — and comparing it against the full MMKG (which required ~2,700 LLM extraction calls to build). The PEG achieved strong factual consistency **without any of those extraction calls**, suggesting construction-cost framing (rather than raw entity-count comparisons) is the more defensible way to present graph-structure ablations.

### D5 — Intermediate Answer Disagreement Signal
An audit of the KG-path vs. naive (image) path in the `mix_two_step` fusion stage found:
- **Mean inter-branch lexical similarity: 0.047** — the two branches' intermediate answers rarely agree in wording.
- **The fusion stage favored image-branch evidence in 79.2% of questions**, despite prompt instructions explicitly telling the model to prefer the knowledge-graph path.

This surfaces a latent disagreement signal in the pipeline that the original fusion prompt does not account for.

### Other structural observations documented during replication
- The `relationship_strength` field is generated by MegaRAG's extraction prompt but is **never stored or used downstream** — a structural gap in the reference implementation.
- All three MegaRAG generation prompts include an inert **timestamp-conflict-resolution block**; the merge step discards the provenance data this logic depends on, so the block never fires in practice.

---

## 🐛 Known Issues and Fixes

| Issue | Fix |
|---|---|
| `IndexError` in refinement when a page has 0 entities | `python debug/apply_refinement_fix.py` |
| `ModuleNotFoundError: omegaconf` in MinerU | `pip install omegaconf` in the `mineru` conda env |
| Relationships extracted with wrong field count (local models) | Set `example_number: 3` in `addon_params.yaml` |
| `0 Ent + 0 Rel` on every chunk | Model not following format — bump `example_number` or use a larger model |
| `PATH` broken after `conda activate` | Re-run `export PATH="$CONDA_PREFIX/bin:$PATH"` after activation (this cluster only) |
| PP-OCR weights fail to load | Remap PP-OCRv5 weight files to PP-OCRv4 paths |
| Ollama unreachable during SLURM build | Start `ollama serve` inside the compute job, not on the login node |

---

## Acknowledgments

MegaRAG is inspired by the work of [LightRAG](https://github.com/HKUDS/LightRAG). This replication and its extensions were carried out as part of the SURAJ 2026 internship at IIT Jodhpur's Machine Intelligence Lab, under the guidance of Dr. Divya Saxena and Aditya Sharma.

## Citation

```bibtex
@inproceedings{hsiao2026megarag,
  title={MegaRAG: Multimodal Graph-based Retrieval Augmented Generation},
  author={Hsiao, Chi-Hsiang and Wang, Yi-Cheng and Lin, Tzung-Sheng and Yeh, Yi-Ren and Chen, Chu-Song},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  url={https://arxiv.org/abs/2512.20626}
}
```

## License

The upstream MegaRAG project is released under a custom license — see its `LICENSE` file for full terms. For academic or commercial use of the base framework, contact the original authors directly.
