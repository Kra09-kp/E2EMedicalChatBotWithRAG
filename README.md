# E2EMedicalChatBotWithRAG


## 🟢 Offline / Low-Data Installation Guide

If you have a slow or limited internet connection but still want to use
`sentence-transformers` (for example to build a RAG pipeline), you can
install everything **offline** and keep it CPU-only to save gigabytes.

### 1️⃣ Download the required wheels on a machine with good internet

Create a folder and download all required packages **without installing**:

```bash
mkdir wheels
cd wheels
# CPU-only PyTorch
pip download torch --index-url https://download.pytorch.org/whl/cpu
# Sentence-Transformers and its Python-only deps
pip download sentence-transformers


then run this cmd after activating your venv
pip install *.whl


you can verify the installation by running this cmd
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print(model.encode('Hello World'))"