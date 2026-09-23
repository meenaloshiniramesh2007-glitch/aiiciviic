"""
TEXT CLASSIFIER TRAINING SCRIPT
================================
AI Civic Guardian - Complaint Category Classifier

This script trains a text classifier that predicts the complaint
category (department) from a user's complaint description.

USAGE:
    python train_text_classifier.py

WHAT IT DOES:
    1. Tries to load Chicago 311 CSV if available
    2. Falls back to built-in sample dataset if CSV not found
    3. Trains TF-IDF + Logistic Regression model
    4. Saves model to models/text_classifier.pkl
    5. Saves label encoder to models/label_encoder.pkl

OUTPUT:
    models/text_classifier.pkl
    models/label_encoder.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "dataset", "text", "311_Service_Requests.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "text_classifier.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── CATEGORY MAPPING ─────────────────────────────────────────────────────────
# Maps raw 311 SR_TYPE values -> our 12 project categories
CATEGORY_MAP = {
    # 1. Road & Pothole
    "Pothole in Street"                         : "Road & Pothole",
    "Pothole"                                   : "Road & Pothole",
    "Crumbling Pavement"                        : "Road & Pothole",
    "Street Sign Repair"                        : "Road & Pothole",
    "Pavement Cave-In Survey"                   : "Road & Pothole",
    # 2. Garbage & Sanitation
    "Sanitation Code Complaint"                 : "Garbage & Sanitation",
    "Overflowing Dumpster"                      : "Garbage & Sanitation",
    "Illegal Dumping"                           : "Garbage & Sanitation",
    "Garbage Cart Maintenance"                  : "Garbage & Sanitation",
    "Recycling Cart Maintenance"                : "Garbage & Sanitation",
    # 3. Water & Drainage
    "Water On Street"                           : "Water & Drainage",
    "Water In Basement"                         : "Water & Drainage",
    "Catch Basin Inspection"                    : "Water & Drainage",
    "Catch Basin Overflow"                      : "Water & Drainage",
    "Sewer Cave-In Inspection"                  : "Water & Drainage",
    "Flooding"                                  : "Water & Drainage",
    "Water Leak"                                : "Water & Drainage",
    "No Water"                                  : "Water & Drainage",
    # 4. Streetlight
    "Street Light - 1 Out"                      : "Streetlight",
    "Street Light All/Out"                      : "Streetlight",
    "Street Light Out"                          : "Streetlight",
    "Lighting Complaint"                        : "Streetlight",
    # 5. Graffiti & Vandalism
    "Graffiti Removal"                          : "Graffiti & Vandalism",
    "Graffiti"                                  : "Graffiti & Vandalism",
    # 6. Building & Infrastructure
    "Building Violation"                        : "Building & Infrastructure",
    "Vacant/Abandoned Building"                 : "Building & Infrastructure",
    "Vacant Lot Clean-Up"                       : "Building & Infrastructure",
    # 7. Noise Complaint
    "Noise - Residential"                       : "Noise Complaint",
    "Noise - Vehicle"                           : "Noise Complaint",
    "Noise - Commercial"                        : "Noise Complaint",
    # 8. Tree & Greenery
    "Tree Debris"                               : "Tree & Greenery",
    "Tree Trim"                                 : "Tree & Greenery",
    "Trees-Damaged Trees"                       : "Tree & Greenery",
    # 9. Air & Pollution
    "Air Quality Complaint"                     : "Air & Pollution",
    "Industrial Pollution"                      : "Air & Pollution",
    "Smoke/Odor Complaint"                      : "Air & Pollution",
    # 10. Animal & Pest
    "Dead Animal Removal"                       : "Animal & Pest",
    "Rodent Baiting/Rat Complaint"              : "Animal & Pest",
    "Animal Complaint"                          : "Animal & Pest",
    "Stray Animal"                              : "Animal & Pest",
    # 11. Traffic & Signals
    "Traffic Signal Out"                        : "Traffic & Signals",
    "Traffic Light Complaint"                   : "Traffic & Signals",
    "Broken Traffic Sign"                       : "Traffic & Signals",
    "Speed Bump Request"                        : "Traffic & Signals",
    # 12. Public Property Damage
    "Park Facility Repair"                      : "Public Property Damage",
    "Public Toilet Complaint"                   : "Public Property Damage",
    "Bench/Bus Shelter Damage"                  : "Public Property Damage",
    "Public Monument Damage"                    : "Public Property Damage",
}

# ─── BUILT-IN SAMPLE DATASET ──────────────────────────────────────────────────
# Used when Chicago 311 CSV is not available.
# You can expand this list as needed.
SAMPLE_DATA = [
    # Road & Pothole
    ("There is a big pothole on the main road near the market", "Road & Pothole"),
    ("Large pothole on MG Road causing accidents", "Road & Pothole"),
    ("Road is damaged and full of potholes near school", "Road & Pothole"),
    ("Deep crater on the highway near flyover", "Road & Pothole"),
    ("Road surface is broken and vehicles are getting stuck", "Road & Pothole"),
    ("Cracked road surface after rain, needs repair urgently", "Road & Pothole"),
    ("Pothole in front of the bus stop is very dangerous", "Road & Pothole"),
    ("Road has several potholes causing two-wheeler accidents", "Road & Pothole"),
    ("Street pavement is crumbling near the junction", "Road & Pothole"),
    ("Damaged road near hospital, emergency vehicles affected", "Road & Pothole"),
    ("Road is uneven and causing damage to vehicles", "Road & Pothole"),
    ("Large hole in road near railway station", "Road & Pothole"),

    # Garbage & Sanitation
    ("Garbage pile not collected for over a week", "Garbage & Sanitation"),
    ("Overflowing garbage bin near the colony entrance", "Garbage & Sanitation"),
    ("Illegal dumping of waste near the park", "Garbage & Sanitation"),
    ("Stinking garbage heap near residential area", "Garbage & Sanitation"),
    ("Waste not collected since 10 days near bus stop", "Garbage & Sanitation"),
    ("Garbage truck has not visited our street in 2 weeks", "Garbage & Sanitation"),
    ("Unauthorized garbage dumping near school", "Garbage & Sanitation"),
    ("Construction waste thrown on public road", "Garbage & Sanitation"),
    ("Open garbage near marketplace causing disease", "Garbage & Sanitation"),
    ("Rubbish piled up near hospital gate", "Garbage & Sanitation"),
    ("Sanitation workers not visiting our area", "Garbage & Sanitation"),
    ("Waste dumped on empty plot near houses", "Garbage & Sanitation"),

    # Water & Drainage
    ("Water pipe leaking on the street for 3 days", "Water & Drainage"),
    ("Drainage blocked causing water logging on road", "Water & Drainage"),
    ("Sewage water overflowing on to the street", "Water & Drainage"),
    ("Water leakage from underground pipe near park", "Water & Drainage"),
    ("Flooding in the basement due to blocked drain", "Water & Drainage"),
    ("No water supply since morning in entire colony", "Water & Drainage"),
    ("Drain is clogged and causing mosquito breeding", "Water & Drainage"),
    ("Sewage smell and overflow near residential houses", "Water & Drainage"),
    ("Waterlogging near school after heavy rain", "Water & Drainage"),
    ("Broken water main causing water to flood road", "Water & Drainage"),
    ("Manhole uncovered causing sewage on road", "Water & Drainage"),
    ("Underground water pipe burst near junction", "Water & Drainage"),

    # Streetlight
    ("Street light not working for 5 days", "Streetlight"),
    ("All lights on our street are out since last week", "Streetlight"),
    ("Streetlight pole is broken and lying on road", "Streetlight"),
    ("No lighting in the park area, unsafe at night", "Streetlight"),
    ("Lamp post near school is not functioning", "Streetlight"),
    ("Multiple street lights are dead in our area", "Streetlight"),
    ("Light flickering and sparking on main road", "Streetlight"),
    ("Dark street causing accidents at night", "Streetlight"),
    ("Streetlight wires hanging dangerously", "Streetlight"),
    ("No street lights working in the entire lane", "Streetlight"),
    ("Light not working near ATM creating unsafe conditions", "Streetlight"),
    ("Public lighting has been off for over a week", "Streetlight"),

    # Graffiti & Vandalism
    ("Graffiti on the government building wall", "Graffiti & Vandalism"),
    ("Vandalism at the public park benches", "Graffiti & Vandalism"),
    ("Someone painted on the compound wall near bus stand", "Graffiti & Vandalism"),
    ("Abusive graffiti on school wall", "Graffiti & Vandalism"),
    ("Public property damaged near market", "Graffiti & Vandalism"),
    ("Defacement of public walls in residential colony", "Graffiti & Vandalism"),

    # Building & Infrastructure
    ("Abandoned building in colony creating safety issue", "Building & Infrastructure"),
    ("Broken footpath tiles causing pedestrian injuries", "Building & Infrastructure"),
    ("Illegal construction blocking the road", "Building & Infrastructure"),
    ("Vacant building used by anti-social elements", "Building & Infrastructure"),
    ("Footpath is damaged and dangerous for walking", "Building & Infrastructure"),
    ("Construction debris blocking the footpath", "Building & Infrastructure"),

    # Noise Complaint
    ("Loud music playing all night from nearby shop", "Noise Complaint"),
    ("Noise from construction site disturbing residents", "Noise Complaint"),
    ("Honking and loud vehicles disturbing peace", "Noise Complaint"),
    ("Loudspeaker noise from event near residential area", "Noise Complaint"),
    ("Factory noise affecting nearby housing area", "Noise Complaint"),
    ("Bar playing loud music beyond permitted hours", "Noise Complaint"),

    # Tree & Greenery
    ("Fallen tree blocking the road after storm", "Tree & Greenery"),
    ("Dead tree branches hanging over road dangerously", "Tree & Greenery"),
    ("Tree overgrown and blocking traffic signal view", "Tree & Greenery"),
    ("Tree roots damaging the road surface", "Tree & Greenery"),
    ("Fallen tree branch on electricity wire", "Tree & Greenery"),
    ("Overgrown tree blocking streetlight", "Tree & Greenery"),

    # Air & Pollution
    ("Black smoke coming from nearby factory", "Air & Pollution"),
    ("Burning garbage producing toxic smoke in our area", "Air & Pollution"),
    ("Industrial waste smell near residential colony", "Air & Pollution"),
    ("Air quality very bad due to construction dust", "Air & Pollution"),
    ("Chemical smell from nearby factory", "Air & Pollution"),
    ("Open burning of waste causing air pollution", "Air & Pollution"),

    # Animal & Pest
    ("Stray dogs attacking people near school", "Animal & Pest"),
    ("Rat infestation in our building colony", "Animal & Pest"),
    ("Dead animal on road not removed for 2 days", "Animal & Pest"),
    ("Stray cattle blocking traffic on main road", "Animal & Pest"),
    ("Cockroach and mosquito problem near drainage", "Animal & Pest"),
    ("Monkeys destroying property in residential area", "Animal & Pest"),

    # Traffic & Signals
    ("Traffic signal not working at main junction", "Traffic & Signals"),
    ("Traffic light broken causing accidents", "Traffic & Signals"),
    ("No speed breaker near school, accidents happening", "Traffic & Signals"),
    ("Road sign fallen and lying on road", "Traffic & Signals"),
    ("No zebra crossing near hospital", "Traffic & Signals"),
    ("Traffic signal timing is wrong causing congestion", "Traffic & Signals"),

    # Public Property Damage
    ("Park bench broken and not repaired for months", "Public Property Damage"),
    ("Public toilet in very bad condition near market", "Public Property Damage"),
    ("Bus shelter damaged, no roof for commuters", "Public Property Damage"),
    ("Statue in public park vandalized", "Public Property Damage"),
    ("Playground equipment broken in colony park", "Public Property Damage"),
    ("Public drinking water tap broken and leaking", "Public Property Damage"),
]


# ─── LOAD OR BUILD DATAFRAME ──────────────────────────────────────────────────
def load_data():
    if os.path.exists(CSV_PATH):
        print(f"[INFO] Loading Chicago 311 CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH, low_memory=False)
        print(f"[INFO] CSV columns: {list(df.columns)}")

        # ── Case 1: Our downloaded CSV (has 'description' and 'category' columns)
        if "description" in df.columns and "category" in df.columns:
            df = df[["description", "category"]].dropna()
            # Merge with built-in samples to cover categories not in the 311 data
            sample_df = build_sample_dataframe()
            df = pd.concat([df, sample_df], ignore_index=True)
            print(f"[INFO] Loaded {len(df)} records (real + sample merged).")
            print(f"[INFO] Category distribution:\n{df['category'].value_counts()}")
            return df

        # ── Case 2: Raw Chicago portal CSV (has 'SR_TYPE' column)
        cat_col = None
        for col in df.columns:
            if col.upper() in ("SR_TYPE", "TYPE", "COMPLAINT_TYPE"):
                cat_col = col
                break

        if cat_col is None:
            print("[WARN] Could not find category column in CSV. Using sample data.")
            return build_sample_dataframe()

        df = df[[cat_col]].dropna()
        df.columns = ["raw_category"]
        df["category"] = df["raw_category"].map(CATEGORY_MAP)
        df = df.dropna(subset=["category"])
        df["description"] = df["raw_category"].str.lower()\
            .str.replace("-", " ").str.replace("/", " ")

        print(f"[INFO] Loaded {len(df)} records from CSV after mapping.")
        print(f"[INFO] Category distribution:\n{df['category'].value_counts()}")

        if len(df) < 50:
            print("[WARN] Too few records. Merging with sample data.")
            sample_df = build_sample_dataframe()
            df = pd.concat([df[["description", "category"]], sample_df], ignore_index=True)

        return df[["description", "category"]]

    else:
        print(f"[INFO] CSV not found at {CSV_PATH}. Using built-in sample dataset.")
        print("[INFO] To use real data, download Chicago 311 CSV and place it at:")
        print(f"       {CSV_PATH}")
        return build_sample_dataframe()


def build_sample_dataframe():
    descriptions = [item[0] for item in SAMPLE_DATA]
    categories   = [item[1] for item in SAMPLE_DATA]
    df = pd.DataFrame({"description": descriptions, "category": categories})
    print(f"[INFO] Sample dataset: {len(df)} records, {df['category'].nunique()} categories")
    return df


# ─── TRAIN ────────────────────────────────────────────────────────────────────
def train():
    print("\n=== AI Civic Guardian - Text Classifier Training ===\n")

    df = load_data()

    X = df["description"].values
    y = df["category"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n[INFO] Classes: {list(le.classes_)}")
    print(f"[INFO] Total samples: {len(X)}")

    # Train/test split
    # Use stratify only if each class has >=2 samples
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

    # Build pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_features=10000,
            sublinear_tf=True,
            min_df=1
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs"
        ))
    ])

    print("\n[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Accuracy: {acc * 100:.2f}%")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model and encoder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"\n[SAVED] Model saved to:   {MODEL_PATH}")
    print(f"[SAVED] Encoder saved to: {ENCODER_PATH}")
    print("\n[DONE] Text classifier training complete!")

    # Quick test
    print("\n--- Quick Prediction Test ---")
    test_texts = [
        "There is a large pothole on the road near school",
        "Garbage not collected for 2 weeks in our colony",
        "Water pipe leaking on main street",
        "Streetlight not working since 5 days",
        "Sewage water overflowing into road",
        "Graffiti on the compound wall near park",
    ]
    for text in test_texts:
        pred_encoded = pipeline.predict([text])[0]
        pred_label   = le.inverse_transform([pred_encoded])[0]
        probs        = pipeline.predict_proba([text])[0]
        confidence   = max(probs) * 100
        print(f"  Input: '{text}'")
        print(f"  → Predicted: {pred_label} ({confidence:.1f}% confidence)\n")


if __name__ == "__main__":
    train()
