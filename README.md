# MegaRAG: Multimodal Graph-based Retrieval Augmented Generation

<p align="center">
  <a href="https://arxiv.org/abs/2512.20626"><img src="https://img.shields.io/badge/arXiv-2512.20626-b31b1b.svg" alt="Paper on ArXiv"></a>
  <img alt="License" src="https://img.shields.io/badge/license-custom-lightgrey">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="ACL 2026" src="https://img.shields.io/badge/ACL-2026-green">
</p>

**MegaRAG** enables **global visual question answering** on documents by constructing a **Multimodal Knowledge Graph (MMKG)**. It combines graph-based reasoning with multimodal page retrieval for precise and rich responses across text and images.

> This work has been accepted to **ACL 2026**.

## 🚀 Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/219ae758-b54c-45c9-a261-b63644ffec08" style="width:90%;" alt="MegaRAG Architecture" />
</p>

MegaRAG builds a knowledge graph from your PDF documents — extracting entities and relationships from both OCR text and page images using a vision LLM — then answers questions by combining graph traversal with multimodal vector search.

## 📦 Installation

### Requirements

* Python 3.10+ (3.10 recommended)
* GPU for embeddings inference (GME model requires CUDA)
* OpenAI-compatible LLM endpoint (OpenAI API key, or local Ollama — see Zero-Cost Setup below)

### Step 1: Install [MinerU](https://github.com/opendatalab/MinerU/tree/release-1.3.6)

```bash
git clone -b release-1.3.6 https://github.com/opendatalab/MinerU.git
cd MinerU

conda create --name mineru python=3.10 -y
conda activate mineru
pip install -e .
pip install omegaconf   # required — not included in MinerU dependencies

pip install huggingface_hub
wget https://raw.githubusercontent.com/opendatalab/MinerU/refs/heads/release-1.3.6/scripts/download_models_hf.py -O download_models_hf.py
sed -i "s|https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json|https://raw.githubusercontent.com/opendatalab/MinerU/release-1.3.6/magic-pdf.template.json|" download_models_hf.py
python download_models_hf.py
```

> **Note:** `omegaconf` must be installed manually in the mineru environment — it is not listed in MinerU's requirements but is required for OCR to work.

### Step 2: Install MegaRAG

```bash
git clone https://github.com/AI-Application-and-Integration-Lab/MegaRAG.git
cd MegaRAG

conda activate mineru
pip install -r requirements_mineru.txt

conda create --name megarag python=3.10 -y
conda activate megarag
pip install -e .

cp .env.sh env.sh
# Fill in your OPENAI_API_KEY and MINERU_PATH in env.sh

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

This patches a known `IndexError` crash in `operate.py` that occurs when any page produces zero entities (cover pages, blank pages, image-only pages). Safe to run multiple times.

## 🆓 Zero-Cost Local Setup (Ollama)

You can run MegaRAG entirely locally using [Ollama](https://ollama.com) instead of the OpenAI API.

**Tested with:** `qwen2.5:7b` for entity extraction and querying, `gme-Qwen2-VL-2B-Instruct` for multimodal embeddings.

```bash
# Install Ollama and pull the model
ollama pull qwen2.5:7b

# Set env.sh to point at Ollama
export OPENAI_API_KEY="ollama"
export OPENAI_API_BASE="http://localhost:11434/v1"

# Start Ollama before running any pipeline step
ollama serve &
```

**Important — set example_number to 3:** When using local models, use 3-shot examples to ensure the model follows the structured output format correctly.

```yaml
# In egs/<your_dataset>/conf/addon_params.yaml
example_number: 3
```

## 🔍 Pre-Flight Validation (run before every build)

MegaRAG includes a debug suite in `debug/` to validate your setup on the login node before submitting GPU jobs. This takes 5 minutes and prevents wasted compute hours.

```bash
conda activate megarag

# Static checks (no GPU, no Ollama needed)
python debug/apply_refinement_fix.py   # patch the IndexError bug
python debug/E_refinement_audit.py     # verify the fix and refinement logic
python debug/B_prompt_inspector.py     # check prompt assembly and image paths

# Live checks (Ollama must be running)
python debug/A_query_smoke_test.py     # imports, LLM call, storage init, keywords

# Post-build validation
python debug/C_graph_inspector.py      # entity count, relationship count, graph health
```

Do not submit a build job until `A_query_smoke_test.py` shows all ✓ PASS.

## ⚡ Quickstart

### 1. Use the Tiny Example

```bash
cd egs/world_history_tiny
mkdir data
```

### 2. Download Example PDF

Download the example PDF and query file from [Google Drive](https://drive.google.com/drive/folders/1iuukUWsxMYobuDRLRJ3dBOkB9mdPGoPp?usp=sharing) and place in `data/`.

### 3. Build the Multimodal Knowledge Graph

```bash
bash ./run_build_mmkg.sh
```

Watch for `Chunk X of N extracted M Ent + K Rel` in the output — M and K should both be > 0. If you see `0 Ent + 0 Rel` on every chunk, increase `example_number` to 3 in `conf/addon_params.yaml`.

### 4. Query with MegaRAG

```bash
bash ./run_querying.sh
```

## 📂 Using Your Own Dataset

### 1. Create a New Recipe

```bash
cp -r egs/.template egs/<your_dataset>
cd egs/<your_dataset>
mkdir data
```

### 2. Add Your Data

Place your PDF in `data/`.

### 3. Edit the Config

```yaml
# conf/addon_params.yaml
example_number: 3          # use 3 for local models; 1 for GPT-4o
language: English
entity_types:
  - your_entity_type_1
  - your_entity_type_2
```

### 4. Build and Query

```bash
bash egs/<your_dataset>/run_build_mmkg.sh
bash egs/<your_dataset>/run_querying.sh
```

## 🏗️ Architecture

MegaRAG extends [LightRAG](https://github.com/HKUDS/LightRAG) with multimodal capabilities:

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
              vdb_chunks.json               PAGE IMAGE vectors (multimodal retrieval)
              graph_*.graphml               knowledge graph
```

**Query pipeline (mix_two_step mode):**
- KG path: keyword extraction → graph traversal → text chunk retrieval → LLM answer
- Naive path: GME vector search on page images → visual LLM answer
- Merge: both answers synthesized into final response

## 🐛 Known Issues and Fixes

| Issue | Fix |
|-------|-----|
| `IndexError` in refinement when a page has 0 entities | Run `python debug/apply_refinement_fix.py` |
| `ModuleNotFoundError: omegaconf` in MinerU | `pip install omegaconf` in the mineru conda env |
| Relationships extracted with wrong field count (local models) | Set `example_number: 3` in addon_params.yaml |
| `0 Ent + 0 Rel` on every chunk | Model not following format — bump example_number or use a larger model |

## Acknowledgments

MegaRAG is inspired by the work of [LightRAG](https://github.com/HKUDS/LightRAG). We are grateful for their excellent tools and contributions.

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

## 📄 License

This project is released under a custom license. See [LICENSE](./LICENSE) for full terms.
For academic or commercial use, please contact the authors directly.
