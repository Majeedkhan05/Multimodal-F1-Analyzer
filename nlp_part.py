"""
nlp_part.py

Hybrid NLP-Heavy TurnNet Inference Script
This module is a standalone single-file script that performs multimodal (image + text)
track difficulty prediction using your existing TurnNet model plus multiple NLP components.

Usage:
    python nlp_part.py

Dependencies:
    pip install torch torchvision pillow sentence-transformers transformers keybert spacy
    python -m spacy download en_core_web_sm

Notes:
- Expects your TurnNet model at: models/turnnet.pt (relative to where you run the script)
- First run will download transformer weights (~ hundreds of MB). Use internet on first run.
- Works on CPU; will be faster with GPU if available.
"""

import os
import sys
import json
import pip

# Force working directory to the folder where this file is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
print(f"[INFO] Working directory set to: {CURRENT_DIR}")

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# -------------------------
# Config / Constants
# -------------------------
MODEL_PATH = os.path.join('models', 'turnnet.pt')  # path to your trained TurnNet model
CLASS_NAMES = ['easy', 'hard']

# Reference sentences for embedding similarity
REF_EASY = "A track with smooth flowing corners, wide runoff areas and easy overtaking"
REF_HARD = "A track with tight narrow corners, frequent chicanes, elevation changes and limited overtaking"

# Fusion weights (tunable)
WEIGHT_IMAGE = 0.4
WEIGHT_ZERO_SHOT = 0.35
WEIGHT_EMBED_SIM = 0.15
WEIGHT_KEYWORD = 0.10

# -------------------------
# TurnNet model definition
# -------------------------
class TurnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Image prediction helper
# -------------------------
def predict_image(image_path, model_path=MODEL_PATH):
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Image not found: {abs_path}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.open(abs_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    model = TurnNet()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TurnNet model not found at {model_path}")

    state = torch.load(model_path, map_location=torch.device('cpu'))
    # support if model was saved as state_dict or full model
    if isinstance(state, dict) and 'net' not in state:
        model.load_state_dict(state)
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            # fallback: assume user accidentally saved the entire model
            model = state
    model.eval()

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().tolist()
        _, predicted = torch.max(outputs, 1)
        label = CLASS_NAMES[predicted.item()]

    return label, probs


# -------------------------
# NLP lazy-loaded state
# -------------------------
_nlp_state = {}

def _safe_import(module_name, pip_name=None):
    try:
        return __import__(module_name)
    except ImportError:
        pkg = pip_name or module_name
        print(f"[Auto-Install] Missing {module_name}, installing {pkg}...")
        pip.main(["install", pkg])
        return __import__(module_name)

def _ensure_sentence_transformer():
    if 'sbert' not in _nlp_state:
        st = _safe_import('sentence_transformers')
        SentenceTransformer = st.SentenceTransformer
        util = st.util
        print("[NLU] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        _nlp_state['sbert'] = SentenceTransformer('all-MiniLM-L6-v2')
        _nlp_state['ref_easy_emb'] = _nlp_state['sbert'].encode(REF_EASY, convert_to_tensor=True)
        _nlp_state['ref_hard_emb'] = _nlp_state['sbert'].encode(REF_HARD, convert_to_tensor=True)
        _nlp_state['util'] = util

def _ensure_zero_shot():
    if 'zero_shot' not in _nlp_state:
        tr = _safe_import('transformers')
        pipeline = tr.pipeline
        print("[NLU] Loading zero-shot classifier (facebook/bart-large-mnli)...")
        _nlp_state['zero_shot'] = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def _ensure_kw_extractor():
    if 'kw' not in _nlp_state:
        _ensure_sentence_transformer()
        kb = _safe_import('keybert')
        KeyBERT = kb.KeyBERT
        print("[NLU] Loading KeyBERT (uses SBERT)...")
        _nlp_state['kw'] = KeyBERT(model=_nlp_state['sbert'])

def _ensure_summarizer():
    if 'summ' not in _nlp_state:
        tr = _safe_import('transformers')
        pipeline = tr.pipeline
        print("[NLU] Loading summarization pipeline (t5-small)...")
        _nlp_state['summ'] = pipeline('summarization', model='t5-small', tokenizer='t5-small')

def _ensure_spacy():
    if 'spacy' not in _nlp_state:
        sp = _safe_import('spacy')
        print("[NLU] Loading spaCy en_core_web_sm...")
        try:
            _nlp_state['spacy'] = sp.load('en_core_web_sm')
        except Exception:
            print("[Auto-Install] Downloading spaCy model en_core_web_sm...")
            pip.main(["install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz"])
            _nlp_state['spacy'] = sp.load('en_core_web_sm')


# -------------------------
# Text analysis helpers
# -------------------------

def text_zero_shot_classify(text):
    _ensure_zero_shot()
    candidate_labels = ['easy', 'hard']
    try:
        result = _nlp_state['zero_shot'](text, candidate_labels)
        top_label = result['labels'][0]
        top_score = float(result['scores'][0])
        return top_label, top_score
    except Exception:
        return 'easy', 0.5


def text_embedding_similarity(text):
    _ensure_sentence_transformer()
    emb = _nlp_state['sbert'].encode(text, convert_to_tensor=True)
    sim_easy = _nlp_state['util'].cos_sim(emb, _nlp_state['ref_easy_emb']).item()
    sim_hard = _nlp_state['util'].cos_sim(emb, _nlp_state['ref_hard_emb']).item()
    if sim_hard > sim_easy:
        return 'hard', float(sim_hard)
    else:
        return 'easy', float(sim_easy)


def extract_keywords(text, top_n=5):
    _ensure_kw_extractor()
    try:
        keywords = _nlp_state['kw'].extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
        return [k[0] for k in keywords]
    except Exception:
        _ensure_spacy()
        doc = _nlp_state['spacy'](text)
        chunks = [chunk.text for chunk in doc.noun_chunks][:top_n]
        return chunks


def summarize_text(text, max_length=60):
    _ensure_summarizer()
    try:
        summary = _nlp_state['summ'](text, max_length=max_length, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception:
        return text if len(text) < 200 else text[:200] + '...'


# -------------------------
# Fusion logic
# -------------------------
def fuse_predictions(image_pred, image_probs, zero_shot_pred, zero_shot_score, emb_pred, emb_score, keywords):
    def label_to_num(l):
        return 1 if l == 'hard' else 0

    img_num = label_to_num(image_pred)
    zs_num = label_to_num(zero_shot_pred)
    emb_num = label_to_num(emb_pred)

    img_conf = float(image_probs[1]) if isinstance(image_probs, (list, tuple)) else 0.5

    weighted = (WEIGHT_IMAGE * img_num * img_conf) + (WEIGHT_ZERO_SHOT * zs_num * zero_shot_score) + (WEIGHT_EMBED_SIM * emb_num * emb_score)

    hard_terms = {'tight', 'sharp', 'chicane', 'elevation', 'narrow', 'limited', 'technical', 'slow', 'hairpin', 'tricky', 'complex'}
    kw_nudge = 0
    for kw in keywords:
        for term in hard_terms:
            if term in kw.lower():
                kw_nudge += 0.1
                break

    final_score = weighted + (WEIGHT_KEYWORD * kw_nudge)
    final_label = 'hard' if final_score >= 0.5 else 'easy'

    reasoning = []
    reasoning.append(f"Image prediction: {image_pred} (hard_prob={img_conf:.2f})")
    reasoning.append(f"Zero-shot text classification: {zero_shot_pred} (score={zero_shot_score:.2f})")
    reasoning.append(f"Embedding similarity: {emb_pred} (score={emb_score:.2f})")
    if keywords:
        reasoning.append(f"Keywords: {', '.join(keywords)}")
    reasoning.append(f"Final numeric fused score: {final_score:.3f} (threshold=0.5)")

    return final_label, '\n'.join(reasoning)


# -------------------------
# Main hybrid function
# -------------------------
def hybrid_predict(image_path, text_description=None, save_report=False):
    try:
        img_label, img_probs = predict_image(image_path)
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

    print(f"\n🖼️ Image-based prediction: {img_label.upper()} | probs={img_probs}")

    if not text_description:
        print('\nNo text description provided. Returning image-only prediction.')
        print(f"Final Decision: {img_label.upper()}")
        return {
            'image_prediction': img_label,
            'image_probs': img_probs,
            'final': img_label
        }

    try:
        zs_label, zs_score = text_zero_shot_classify(text_description)
    except Exception as e:
        print(f"Zero-shot classification failed: {e}")
        zs_label, zs_score = 'easy', 0.5

    try:
        emb_label, emb_score = text_embedding_similarity(text_description)
    except Exception as e:
        print(f"Embedding similarity failed: {e}")
        emb_label, emb_score = 'easy', 0.5

    try:
        keywords = extract_keywords(text_description, top_n=6)
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        keywords = []

    try:
        summary = summarize_text(text_description)
    except Exception as e:
        print(f"Summarization failed: {e}")
        summary = text_description[:200]

    print(f"\n🗣️ Zero-shot text classification: {zs_label.upper()} (score={zs_score:.2f})")
    print(f"🧾 Embedding-based similarity: {emb_label.upper()} (score={emb_score:.2f})")
    print(f"🔍 Keywords: {keywords}")
    print(f"✂️ Text summary: {summary}")

    final_label, reasoning = fuse_predictions(img_label, img_probs, zs_label, zs_score, emb_label, emb_score, keywords)

    expl = f"Based on the image prediction of {img_label.upper()} and text signals (zero-shot: {zs_label.upper()}, embedding: {emb_label.upper()}), the system predicts: {final_label.upper()}."

    print(f"\n🧠 Final Decision: {final_label.upper()}\n")
    print("Reasoning breakdown:\n")
    print(reasoning)
    print("\nNatural language explanation:\n")
    print(expl)

    result = {
        'image_prediction': img_label,
        'image_probs': img_probs,
        'zero_shot': (zs_label, zs_score),
        'embedding': (emb_label, emb_score),
        'keywords': keywords,
        'summary': summary,
        'final': final_label,
        'reasoning': reasoning,
        'explanation': expl
    }

    if save_report:
        try:
            with open('nlp_prediction_report.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("Saved report to nlp_prediction_report.json")
        except Exception as e:
            print(f"Failed to save report: {e}")

    return result


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == '__main__':
    print("Hybrid TurnNet + NLP Inference - nlp_part.py")
    image_path = input("Enter track image path: ").strip()
    if image_path == '':
        print("No image path provided. Exiting.")
        sys.exit(0)

    text_description = input("Enter a textual description/commentary about the track (optional): ").strip()
    save_choice = input("Save JSON report? (y/N): ").strip().lower()
    save_report = True if save_choice == 'y' else False

    hybrid_predict(image_path, text_description if text_description else None, save_report=save_report)

    print("\nDone.")
