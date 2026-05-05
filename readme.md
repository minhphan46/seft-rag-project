# SEFT RAG Project

Project nay dung de chay va thu nghiem `self-rag` voi checkpoint local `selfrag_llama2_7b` theo paper Self-RAG.

## Cau truc thu muc

- `self-rag/`: ma nguon goc Self-RAG (inference, retrieval, training scripts)
- `selfrag_llama2_7b/`: checkpoint model local (config, tokenizer, model shards)
- `self-rag/quick_test.py`: script test suy luan nhanh tren may local

## Tai lieu tham khao

- Paper: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- Model card: [selfrag/selfrag_llama2_7b](https://huggingface.co/selfrag/selfrag_llama2_7b)
- Repo goc: [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)

## Yeu cau toi thieu

- Python 3.10
- RAM/VRAM du lon cho model 7B (neu chay CPU se rat cham)
- Da co day du 2 file weights trong `selfrag_llama2_7b`:
  - `pytorch_model-00001-of-00002.bin`
  - `pytorch_model-00002-of-00002.bin`

## Huong dan chay nhanh (local quick test)

Tu thu muc `self-rag`:

```powershell
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install transformers torch sentencepiece accelerate protobuf tiktoken
py -3.10 quick_test.py
```

Neu thay dong `Generating...`, script dang sinh token (co the mat nhieu thoi gian tren CPU/offload).

## Chay theo pipeline Self-RAG trong paper

1. Cai dependencies day du trong `self-rag/requirements.txt`.
2. Chuan bi retrieval data theo README cua repo goc (`retrieval_lm/download_demo_corpus.sh` hoac corpus rieng).
3. Chay short-form evaluation:

```bash
python run_short_form.py \
  --model_name /path/to/selfrag_llama2_7b \
  --input_file eval_data/popqa_longtail_w_gs.jsonl \
  --mode adaptive_retrieval \
  --max_new_tokens 100 \
  --threshold 0.2 \
  --output_file ./out_popqa.jsonl \
  --metric match --ndocs 10 \
  --use_groundness --use_utility --use_seqscore \
  --dtype half
```

`adaptive_retrieval` la mode gan dung setup trong paper cho viec quyet dinh retrieve theo reflection tokens.

## Luu y quan trong tren Windows

- `vllm`/`flash-attn` thuong on dinh hon tren Linux/WSL2 + CUDA.
- Neu chay native Windows va khong co GPU phu hop, toc do se cham do offload ra dia.
- Neu can benchmark gan paper, uu tien Linux + GPU.

## Buoc tiep theo (chi tiet)

Tot, ban da vao duoc `py -3.10`. Hay lam theo dung thu tu ben duoi:

### 1) Thoat Python REPL ve PowerShell

Neu dang thay dau `>>>`, chay:

```python
exit()
```

### 2) Tao virtual env rieng va kich hoat

Trong thu muc `self-rag`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Cai dependencies toi thieu de test model local

Tren Windows, `vllm`/`flash-attn` thuong kho cai. Truoc mat test bang `transformers`:

```powershell
pip install torch transformers sentencepiece accelerate
```

Neu gap loi tokenizer, cai them:

```powershell
pip install protobuf tiktoken
```

### 4) Kiem tra model local da du weight chua

Vao thu muc model `selfrag_llama2_7b`, can co:

- `pytorch_model-00001-of-00002.bin`
- `pytorch_model-00002-of-00002.bin`

Neu chua co thi pull LFS:

```powershell
git lfs install
git lfs pull
```

### 5) Chay test suy luan nhanh (khong retrieval)

Tao file `quick_test.py` trong `self-rag`, roi chay:

```powershell
py -3.10 quick_test.py
```

Neu thay dong `Generating...`, model dang sinh token va khong bi treo.
Tren CPU/offload, buoc nay co the rat cham.

---

Neu muon chay dung pipeline paper (`run_short_form.py`, `adaptive_retrieval`), xem muc `Chay theo pipeline Self-RAG trong paper` o tren va dung lenh mau tai do.
