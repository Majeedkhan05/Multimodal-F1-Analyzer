# 🏎️ F1 Turn Difficulty Classifier — Hybrid Computer Vision + NLP Model

## 📌 Overview
This project began as a pure **CNN-based image classifier** for predicting whether an F1 track turn is **Easy** or **Hard** based on its layout image.

I later upgraded it into a **multimodal AI system** that combines:
- 🖼 **Computer Vision (CNN / TurnNet)**
- 🧠 **Natural Language Processing (Transformer-based models)**
- 🔗 **Fusion Logic** to combine predictions from CV + NLP

This makes the project much more powerful, research-like, and demo-ready.

---

# 🚀 Features

### ✅ **1. Image-Based Difficulty Prediction (CNN)**
- Custom CNN model called **TurnNet**
- Trained on a manually labeled dataset of F1 track turn images
- Predicts: **Easy** or **Hard**

### ✅ **2. NLP-Based Analysis (Multi-Model Pipeline)**
When the user enters a text description of the track (optional), the system performs:

- **Zero-shot text classification** (BART-MNLI)
- **Embedding similarity** using Sentence-BERT
- **Keyword extraction** using KeyBERT
- **Summarization** using T5
- **spaCy** for fallback NLP processing

### ✅ **3. Multimodal Fusion Logic**
The final decision is computed by combining:
- CNN image probability
- Zero-shot score
- SBERT similarity score
- Keyword difficulty indicators

This produces:
- ✔ A **final difficulty label**
- ✔ A **detailed reasoning breakdown**
- ✔ A **natural language explanation**
- ✔ An optional **JSON report**

---

# 🧠 Tech Stack

### **Computer Vision:**
- PyTorch
- Torchvision
- Custom CNN (TurnNet)

### **NLP:**
- SentenceTransformer (SBERT)
- Transformers (BART, T5)
- KeyBERT
- spaCy

### **Others:**
- Python
- JSON report generation

---

# 📂 Project Structure
```
F1_Turn_Detector/
│
├── f1env/                     # Virtual environment
│   ├── nlp_part.py           # Hybrid CV + NLP inference script
│
├── models/
│   └── turnnet.pt            # Trained CNN model
│
├── turn_dataset/             # Image dataset
│   ├── easy/
│   └── hard/
│
└── README.md
```

---

# 🧪 How to Run the Hybrid Model

### **1. Activate environment**
```bash
source f1env/bin/activate
```

### **2. Run script**
```bash
cd f1env
python nlp_part.py
```

### **3. Enter image path**
You can use relative paths such as:
```
../turn_dataset/easy/monaco_01.png
```
Or drag-drop an image into the terminal.

### **4. (Optional) Enter text description**
Example:
```
This track has very tight turns and a difficult hairpin.
```
This activates all NLP models.

### **5. Save JSON report?**
Type:
```
y
```
A report is generated as `nlp_prediction_report.json`.

---

# 📊 Example Output
```
🖼️ Image-based prediction: EASY | probs=[0.9997, 0.0002]
🗣️ Zero-shot text classification: HARD (score=0.87)
🧾 Embedding similarity: HARD (score=0.81)
🔍 Keywords: ["tight turns", "hairpin"]
✂️ Text summary: "The track contains tight and technical corners ..."

🧠 Final Decision: HARD
```

---

# 🎯 What I Learned
- Designing and training CNNs for image classification
- Building hybrid **multimodal** AI systems
- Using multiple Transformer-based NLP models in a single pipeline
- Fusing CV + NLP predictions
- Dataset balancing, augmentation, and preprocessing
- Creating user-friendly interactive inference scripts

---

# 🏁 Future Improvements
- Build a Streamlit UI
- Support full-track difficulty instead of turn-level
- Add attention heatmaps for CNN interpretation (GradCAM)
- Try CLIP for joint vision-language classification

---

# 🙌 Acknowledgements
This project was created to combine my love for **F1 racing** with my interest in **AI and Deep Learning**.