"""
CLASSIFIER HELPER MODULE
=========================
AI Civic Guardian

This module is imported by app.py to provide:
  1. predict_category(text)  → predicts complaint department from description
  2. predict_image(filepath) → predicts civic issue type from an image file

Both functions are safe — they return a fallback value if the model
file is missing (model not trained yet), so app.py won't crash.

USAGE IN app.py:
    from classifier import predict_category, predict_image

    # Text prediction
    dept = predict_category("There is a large pothole near the school")
    # Returns: "Road & Pothole"

    # Image prediction
    result = predict_image("static/uploads/complaint_001.jpg")
    # Returns: {"label": "pothole", "confidence": 87.3}
"""

import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, "models")

TEXT_MODEL_PATH   = os.path.join(MODELS_DIR, "text_classifier.pkl")
TEXT_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

IMG_MODEL_PATH    = os.path.join(MODELS_DIR, "image_classifier.h5")
IMG_CLASSES_PATH  = os.path.join(MODELS_DIR, "image_classifier.pkl")

IMG_SIZE = (128, 128)

# ─── LAZY LOADING ─────────────────────────────────────────────────────────────
# Models are loaded once on first use, not at import time.
# This avoids slowing down Flask startup.
_text_model   = None
_text_encoder = None
_img_model    = None
_img_classes  = None


def _load_text_model():
    global _text_model, _text_encoder
    if _text_model is not None:
        return True
    try:
        with open(TEXT_MODEL_PATH, "rb") as f:
            _text_model = pickle.load(f)
        with open(TEXT_ENCODER_PATH, "rb") as f:
            _text_encoder = pickle.load(f)
        print("[Classifier] Text model loaded successfully.")
        return True
    except FileNotFoundError:
        print("[Classifier] Text model not found. Run train_text_classifier.py first.")
        return False
    except Exception as e:
        print(f"[Classifier] Error loading text model: {e}")
        return False


def _load_img_model():
    global _img_model, _img_classes
    if _img_model is not None:
        return True
    try:
        import tensorflow as tf
        _img_model = tf.keras.models.load_model(IMG_MODEL_PATH)
        with open(IMG_CLASSES_PATH, "rb") as f:
            _img_classes = pickle.load(f)
        print("[Classifier] Image model loaded successfully.")
        return True
    except FileNotFoundError:
        print("[Classifier] Image model not found. Run train_image_classifier.py first.")
        return False
    except ImportError:
        print("[Classifier] TensorFlow not installed. Image prediction unavailable.")
        return False
    except Exception as e:
        print(f"[Classifier] Error loading image model: {e}")
        return False


# ─── TEXT PREDICTION ──────────────────────────────────────────────────────────
# Maps predicted label → department name shown in the UI
LABEL_TO_DEPARTMENT = {
    "Road & Pothole"           : "Roads & Infrastructure Dept",
    "Garbage & Sanitation"     : "Sanitation & Waste Dept",
    "Water & Drainage"         : "Water Supply & Sewage Board",
    "Streetlight"              : "Electricity & Lighting Dept",
    "Graffiti & Vandalism"     : "Public Works Dept",
    "Building & Infrastructure": "Building & Planning Dept",
    "Noise Complaint"          : "Municipal Corporation",
    "Tree & Greenery"          : "Horticulture & Parks Dept",
    "Air & Pollution"          : "Environment & Pollution Dept",
    "Animal & Pest"            : "Animal Control & Health Dept",
    "Traffic & Signals"        : "Traffic Engineering Dept",
    "Public Property Damage"   : "Civic Maintenance Dept",
}

def predict_category(text):
    """
    Predict complaint category from description text.

    Args:
        text (str): User's complaint description

    Returns:
        dict: {
            "category"   : "Road & Pothole",
            "department" : "Roads & Infrastructure",
            "confidence" : 91.5,
            "available"  : True
        }
        On failure:
        {
            "category"   : "General",
            "department" : "Municipal Corporation",
            "confidence" : 0,
            "available"  : False
        }
    """
    fallback = {
        "category"  : "General",
        "department": "Municipal Corporation",
        "confidence": 0,
        "available" : False
    }

    if not text or not text.strip():
        return fallback

    if not _load_text_model():
        return fallback

    try:
        text_clean = text.strip().lower()
        pred_encoded = _text_model.predict([text_clean])[0]
        pred_label   = _text_encoder.inverse_transform([pred_encoded])[0]
        probs        = _text_model.predict_proba([text_clean])[0]
        confidence   = round(float(max(probs)) * 100, 1)
        department   = LABEL_TO_DEPARTMENT.get(pred_label, "Municipal Corporation")

        return {
            "category"  : pred_label,
            "department": department,
            "confidence": confidence,
            "available" : True
        }
    except Exception as e:
        print(f"[Classifier] Text prediction error: {e}")
        return fallback


def predict_category_simple(text):
    """
    Simplified version — returns just the department string.
    Useful for auto-filling the complaint form dropdown.

    Returns:
        str: department name like "Roads & Infrastructure"
    """
    result = predict_category(text)
    return result["department"]


# ─── IMAGE PREDICTION ─────────────────────────────────────────────────────────

# Maps image class name → human-readable label for UI display
IMAGE_LABEL_DISPLAY = {
    "pothole"        : "Pothole / Road Damage",
    "garbage"        : "Garbage / Waste Accumulation",
    "water"          : "Water Leakage / Flooding",
    "streetlight"    : "Streetlight Issue",
    "drainage"       : "Drainage / Sewage Problem",
    "graffiti"       : "Graffiti / Vandalism",
    "air_pollution"  : "Air Pollution / Smoke",
    "animal_pest"    : "Animal / Pest Problem",
    "traffic_signal" : "Traffic Signal / Road Sign Issue",
    "building_damage": "Building / Infrastructure Damage",
    "tree_fallen"    : "Fallen Tree / Branch",
    "normal"         : "No Visible Civic Issue",
}

def predict_image(image_path):
    """
    Predict the civic issue type from an image file.

    Args:
        image_path (str): Full or relative path to the image file

    Returns:
        dict: {
            "label"      : "pothole",
            "display"    : "Pothole / Road Damage",
            "confidence" : 87.3,
            "all_scores" : {"pothole": 87.3, "garbage": 5.1, ...},
            "available"  : True
        }
        On failure:
        {
            "label"      : "unknown",
            "display"    : "Unable to analyse image",
            "confidence" : 0,
            "all_scores" : {},
            "available"  : False
        }
    """
    fallback = {
        "label"     : "unknown",
        "display"   : "Unable to analyse image",
        "confidence": 0,
        "all_scores": {},
        "available" : False
    }

    if not image_path or not os.path.exists(image_path):
        print(f"[Classifier] Image not found: {image_path}")
        return fallback

    if not _load_img_model():
        return fallback

    try:
        from PIL import Image

        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # shape: (1, 128, 128, 3)

        # Predict
        probs = _img_model.predict(arr, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_label = _img_classes[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 1)
        display    = IMAGE_LABEL_DISPLAY.get(pred_label, pred_label.title())

        # All scores
        all_scores = {
            _img_classes[i]: round(float(probs[i]) * 100, 1)
            for i in range(len(_img_classes))
        }

        return {
            "label"     : pred_label,
            "display"   : display,
            "confidence": confidence,
            "all_scores": all_scores,
            "available" : True
        }

    except ImportError:
        print("[Classifier] Pillow not installed. Run: pip install Pillow")
        return fallback
    except Exception as e:
        print(f"[Classifier] Image prediction error: {e}")
        return fallback


# ─── STATUS HELPERS ───────────────────────────────────────────────────────────

def get_classifier_status():
    """
    Returns status of both classifiers.
    Useful for showing status in admin dashboard.

    Returns:
        dict: {
            "text_model_ready" : True/False,
            "image_model_ready": True/False,
        }
    """
    text_ready  = os.path.exists(TEXT_MODEL_PATH) and os.path.exists(TEXT_ENCODER_PATH)
    image_ready = os.path.exists(IMG_MODEL_PATH)  and os.path.exists(IMG_CLASSES_PATH)
    return {
        "text_model_ready" : text_ready,
        "image_model_ready": image_ready,
    }


# ─── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== classifier.py — Quick Test ===\n")

    print("--- Text Classification ---")
    test_descriptions = [
        "There is a huge pothole on the road near the bus stop",
        "Garbage has been piling up for 2 weeks in our area",
        "Water pipe burst near the park, water flooding the road",
        "Street light not working since last week",
        "Sewage water overflowing near our colony",
    ]
    for desc in test_descriptions:
        result = predict_category(desc)
        if result["available"]:
            print(f"  '{desc[:50]}...'")
            print(f"   → {result['category']} | {result['department']} | {result['confidence']}%\n")
        else:
            print(f"  [Model not trained] Fallback: {result['department']}\n")

    print("--- Classifier Status ---")
    status = get_classifier_status()
    print(f"  Text model ready : {status['text_model_ready']}")
    print(f"  Image model ready: {status['image_model_ready']}")
