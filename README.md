# Daily Papers

Daily Papers is a small LangGraph pipeline for collecting recent research papers, downloading open-access PDFs, and generating daily deep-dive study notes with an LLM.

The main runnable workflow is `main_graph.py`.

## What It Does

1. Searches Semantic Scholar for recent papers from a fixed query list.
2. Downloads open-access PDFs into `ai_knowledge_base/`.
3. Skips papers already recorded in `read_history.json`.
4. Extracts text from the first two pages with PyMuPDF.
5. Sends the extracted text to SiliconFlow's OpenAI-compatible API using `deepseek-ai/DeepSeek-V3`.
6. Writes Markdown study notes into `Study_Notes/`.

## Files

- `main_graph.py`: production pipeline entry point.
- `paper_reading.py`: Semantic Scholar search and PDF download helpers.
- `agentstate.py`: experimental LangGraph state/checkpoint example with manual review.
- `langgraph_pipeline.py`: simple scout-node example that calls `paper_reading.fetch_papers`.
- `pdf_parser.py`: optional Nougat OCR helper for formula-heavy PDF parsing.
- `run.sh`: setup and run script for the main pipeline.

## Requirements

- Python 3.10 or newer is recommended.
- A SiliconFlow API key is required for LLM analysis.
- A Semantic Scholar API key is optional but recommended to reduce rate limits.

The runner installs these Python packages into a local `.venv`:

- `requests`
- `PyMuPDF`
- `pydantic`
- `langgraph`
- `langchain-openai`

## Setup

Set the required API key:

```bash
export SILICONFLOW_API_KEY="your_siliconflow_key"
```

Optionally set a Semantic Scholar API key:

```bash
export S2_API_KEY="your_semantic_scholar_key"
```

## Run

```bash
chmod +x run.sh
./run.sh
```

The first run creates `.venv` and installs dependencies. Later runs reuse the same virtual environment.

## Outputs

- `ai_knowledge_base/`: downloaded PDFs from the main pipeline.
- `Study_Notes/Deep_Dive_YYYY-MM-DD.md`: generated daily notes.
- `read_history.json`: local list of Semantic Scholar paper IDs that have already been processed.

## Notes

- `main_graph.py` currently uses this fixed search matrix:
  - `Large Language Model reasoning paths`
  - `Multi-Agent reinforcement learning collaboration`
  - `Energy-based models diffusion`
  - `AI for Science Physics foundation models`
- Each query fetches up to two new papers, so a full run can analyze up to eight papers.
- LLM calls may incur API costs.
- `pdf_parser.py` requires the external `nougat` command if you choose to use it directly; the main pipeline does not call it.
