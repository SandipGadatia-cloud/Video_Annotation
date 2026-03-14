# 🎬 Gemini + SAM Video Annotation Agent

An AI-powered video annotation system that uses **Gemini Vision + Segment Anything Model (SAM)** to detect objects, segment them, analyze relationships, and generate an ontology/knowledge graph.

## 🚀 Features

* Video frame extraction
* Gemini Vision object detection
* SAM object segmentation
* Object relationship analysis
* Automatic ontology generation
* Knowledge graph visualization
* Human-in-the-loop feedback
* LangGraph agent workflow
* Interactive Streamlit UI

---

## 📦 Installation

Clone the repository:

```
git clone https://github.com/SandipGadatia-cloud/Video_Annotation.git
cd Video_Annotation
```

Create a virtual environment:

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root.

```
GOOGLE_API_KEY=your_google_api_key
```

---

## 🧠 Download SAM Model

Download the SAM model:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Place it in the project root.

---

## ▶️ Run the Application

```
streamlit run app.py
```

---

## 🧩 Tech Stack

* Streamlit
* LangGraph
* Gemini Vision API
* Segment Anything Model (SAM)
* OpenCV
* NumPy
* Python

---

## 👥 Contributors

* Sandip
