# 🚀 **Project Setup**

This project is developed on **🐍 Python 3.12** — download it from the **Microsoft Store**.
It uses **Streamlit** and a few external libraries that must be installed before running.

After setup, you may launch the app using **`run_app.bat`** in the main program folder.

---

## 🧩 **1. Create and Activate a Virtual Environment (Recommended)**

```bash
# 💻 Open command line in the project folder
Right-click any blank space → "Open in Terminal"

# 🐍 Create a virtual environment (Recommended)
python -m venv venv

# ⚡ Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

> 💡 *You may skip this step if you prefer using your global Python environment.*

---

## 📦 **2. Install Required Packages**

```bash
pip install -r requirements.txt --verbose --progress-bar=on
```

---

## 🔑 **3. Setup / Create `.env` File (API_KEY)**

```bash
# 🧾 Create a .env file in the main folder
# Fill in your API key — you can refer to the sample provided in `.env_sample`
```

---

## ▶️ **4. Run the Project**


Simply **double-click**
# `rag_app.bat` in the main folder.
The Streamlit app should open automatically in your browser.

> If not, check your terminal for a link (usually `http://localhost:8501`).


### 💡 **4.1 Alternative Run Method**

```bash
# If virtual environment activated
streamlit run rag_main.py

# OR if not using virtual environment
python -m streamlit run rag_main.py
```

---

## 📚 **5. Upload Knowledge Base (‘kb’ Folder)**

1. 🗂️ Place your notes or documents into the **`kb/`** folder.
2. ⚙️ From the Streamlit sidebar, click **“Ingest KB”**.
3. ⏳ Wait until the ingestion process completes.

A **`chroma_db/`** folder will be automatically created — it stores embeddings of your notes.

> 💬 **Tip:** Include the document name inside your text to **improve search accuracy**.

---

## 🔍 **6. Query Using Natural Language**

Once your knowledge is ingested, simply type a natural-language question.
The system retrieves, reranks, and optionally uses an **LLM** to generate answers.

---

## ⚡ **Quick Commands**

### ▶️ **Run from Command Line**

```bash
python -m streamlit run rag_main.py
```

### 🧠 **Activate Virtual Environment**

```bash
# On Windows (PowerShell):
.\venv\Scripts\activate

# On Linux / macOS:
source venv/bin/activate
```

---

### 🗑️ **Reset Tip**

> To reset the database, delete all contents in the **`chroma_db/`** folder.

---
