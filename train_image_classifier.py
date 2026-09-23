"""
IMAGE CLASSIFIER TRAINING SCRIPT
==================================
AI Civic Guardian - Civic Issue Image Classifier

This script trains a CNN image classifier that predicts the type of
civic issue from an uploaded complaint image.

SUPPORTED CATEGORIES (based on folder names in dataset/images/):
    - pothole      (road damage)
    - garbage      (waste accumulation)
    - water        (water leakage / flooding)
    - streetlight  (broken streetlight)
    - drainage     (blocked drain / sewage)
    - normal       (no visible civic issue)

DATASET OPTIONS:
    OPTION A - Use your own downloaded images:
        Place images in dataset/images/<category>/ folders.
        Minimum ~30 images per category recommended.

    OPTION B - No images yet:
        Run with --demo flag to generate synthetic colored placeholder images.
        This lets you verify the pipeline works before real images.
        python train_image_classifier.py --demo

USAGE:
    python train_image_classifier.py           # train on real images
    python train_image_classifier.py --demo    # generate demo images + train

OUTPUT:
    models/image_classifier.h5       (Keras model)
    models/image_classifier.pkl      (class names list)
    models/image_classifier_info.txt (training summary)
"""

import os
import sys
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR     = os.path.join(BASE_DIR, "dataset", "images")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
MODEL_H5_PATH  = os.path.join(MODELS_DIR, "image_classifier.h5")
CLASSES_PATH   = os.path.join(MODELS_DIR, "image_classifier.pkl")
INFO_PATH      = os.path.join(MODELS_DIR, "image_classifier_info.txt")

os.makedirs(MODELS_DIR, exist_ok=True)

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
IMG_SIZE    = (128, 128)   # resize all images to this
BATCH_SIZE  = 16
EPOCHS      = 20           # increase to 40+ for better accuracy with real data
LEARNING_RATE = 0.001

# ─── DEMO IMAGE GENERATOR ─────────────────────────────────────────────────────
# Creates simple synthetic colored images for each category.
# Only used when --demo flag is passed or no real images exist.
DEMO_CATEGORIES = {
    "pothole":        (80,  80,  80),   # dark grey      – road surface
    "garbage":        (100, 120, 60),   # olive green    – waste
    "water":          (50,  120, 200),  # blue           – water / flooding
    "streetlight":    (240, 220, 80),   # yellow         – broken light
    "drainage":       (60,  80,  100),  # dark blue-grey – sewage / drain
    "graffiti":       (200, 80,  120),  # pink-red       – spray paint
    "air_pollution":  (160, 140, 100),  # smoky brown    – smog / smoke
    "animal_pest":    (140, 100, 60),   # brown          – animals / pests
    "traffic_signal": (220, 60,  60),   # red            – broken signal
    "building_damage":(180, 160, 130),  # beige          – cracked wall
    "tree_fallen":    (60,  130, 60),   # green          – fallen tree
    "normal":         (180, 200, 180),  # light green    – clean / no issue
}
DEMO_IMAGES_PER_CLASS = 60  # images to generate per category


def generate_demo_images():
    """Generate synthetic colored images for demo/testing purposes."""
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import random
    except ImportError:
        print("[ERROR] Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    print("\n[DEMO] Generating synthetic demo images...")

    for category, base_color in DEMO_CATEGORIES.items():
        cat_dir = os.path.join(IMAGES_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)

        # Skip if already has real images
        existing = [f for f in os.listdir(cat_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(existing) >= DEMO_IMAGES_PER_CLASS:
            print(f"  [SKIP] {category}: {len(existing)} images already exist")
            continue

        for i in range(DEMO_IMAGES_PER_CLASS):
            # Create base colored image with noise
            img_array = np.zeros((128, 128, 3), dtype=np.uint8)
            for c in range(3):
                noise = np.random.randint(-40, 40, (128, 128))
                img_array[:, :, c] = np.clip(base_color[c] + noise, 0, 255)

            img = Image.fromarray(img_array)

            # Add some texture variation
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

            # Add category-specific pattern
            draw = ImageDraw.Draw(img)
            r, g, b = base_color

            if category == "pothole":
                # Draw dark circular potholes
                for _ in range(random.randint(1, 4)):
                    x, y = random.randint(10, 118), random.randint(10, 118)
                    size  = random.randint(8, 25)
                    draw.ellipse([x, y, x+size, y+size], fill=(30, 30, 30))

            elif category == "garbage":
                # Draw messy rectangles
                for _ in range(random.randint(5, 15)):
                    x, y = random.randint(0, 110), random.randint(0, 110)
                    draw.rectangle([x, y, x+random.randint(5,20), y+random.randint(5,15)],
                                   fill=(random.randint(50,150),
                                         random.randint(50,120),
                                         random.randint(10,60)))

            elif category == "water":
                # Draw horizontal water ripple lines
                for _ in range(random.randint(3, 8)):
                    y = random.randint(10, 120)
                    draw.line([(0, y), (128, y+random.randint(-3,3))],
                              fill=(20, 80, 200), width=random.randint(1,3))

            elif category == "streetlight":
                # Draw a pole and light glow
                draw.rectangle([60, 20, 68, 128], fill=(80, 80, 80))
                draw.ellipse([40, 5, 88, 45], fill=(255, 255, 150))

            elif category == "drainage":
                # Draw grid pattern (manhole/grate)
                for row in range(0, 128, 12):
                    draw.line([(0, row), (128, row)], fill=(30, 50, 70), width=1)
                for col in range(0, 128, 12):
                    draw.line([(col, 0), (col, 128)], fill=(30, 50, 70), width=1)

            elif category == "graffiti":
                # Draw random colored strokes
                for _ in range(random.randint(3, 8)):
                    x1, y1 = random.randint(0, 100), random.randint(0, 100)
                    x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(5, 30)
                    col = (random.randint(150, 255), random.randint(0, 100), random.randint(50, 200))
                    draw.line([(x1, y1), (x2, y2)], fill=col, width=random.randint(2, 6))

            elif category == "air_pollution":
                # Draw hazy horizontal bands
                for row in range(0, 128, 8):
                    alpha = random.randint(100, 180)
                    draw.rectangle([0, row, 128, row+6],
                                   fill=(alpha, alpha-20, alpha-40))

            elif category == "animal_pest":
                # Draw small oval shapes (animals/pests)
                for _ in range(random.randint(2, 6)):
                    x, y = random.randint(5, 100), random.randint(5, 100)
                    draw.ellipse([x, y, x+random.randint(10,25), y+random.randint(8,18)],
                                 fill=(100, 70, 40))

            elif category == "traffic_signal":
                # Draw traffic signal pole and lights
                draw.rectangle([58, 30, 70, 128], fill=(60, 60, 60))
                draw.ellipse([50, 10, 78, 35], fill=(220, 30, 30))   # red
                draw.ellipse([50, 38, 78, 63], fill=(220, 180, 30))  # yellow
                draw.ellipse([50, 66, 78, 91], fill=(30, 180, 30))   # green

            elif category == "building_damage":
                # Draw crack lines on wall
                for _ in range(random.randint(2, 5)):
                    x, y = random.randint(10, 80), random.randint(5, 60)
                    pts = [(x, y)]
                    for _ in range(random.randint(3, 7)):
                        x += random.randint(-10, 10)
                        y += random.randint(5, 20)
                        pts.append((max(0, min(127, x)), max(0, min(127, y))))
                    draw.line(pts, fill=(50, 40, 30), width=2)

            elif category == "tree_fallen":
                # Draw a diagonal log/tree trunk
                draw.line([(0, 40), (128, 90)], fill=(80, 50, 20), width=12)
                # Some leaves/branches
                for _ in range(random.randint(5, 12)):
                    x, y = random.randint(10, 120), random.randint(20, 80)
                    draw.ellipse([x, y, x+random.randint(8,20), y+random.randint(6,15)],
                                 fill=(30+random.randint(0,60), 100+random.randint(0,60), 20))

            # Save
            fname = os.path.join(cat_dir, f"demo_{category}_{i:04d}.jpg")
            img.save(fname, "JPEG", quality=85)

        print(f"  [OK] {category}: {DEMO_IMAGES_PER_CLASS} demo images created")

    print("[DEMO] Demo image generation complete.\n")


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_dataset():
    """Load images from dataset/images/<category>/ folders."""
    try:
        from PIL import Image
    except ImportError:
        print("[ERROR] Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    categories = sorted([
        d for d in os.listdir(IMAGES_DIR)
        if os.path.isdir(os.path.join(IMAGES_DIR, d))
    ])

    if not categories:
        print(f"[ERROR] No category folders found in {IMAGES_DIR}")
        print("  Create subfolders like: pothole/, garbage/, water/, etc.")
        sys.exit(1)

    print(f"[INFO] Found categories: {categories}")

    X, y = [], []
    skipped = 0

    for label_idx, category in enumerate(categories):
        cat_dir = os.path.join(IMAGES_DIR, category)
        image_files = [
            f for f in os.listdir(cat_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        if len(image_files) == 0:
            print(f"  [WARN] {category}: 0 images — skipping this category")
            continue

        print(f"  Loading {category}: {len(image_files)} images...", end=" ")
        loaded = 0

        for fname in image_files:
            fpath = os.path.join(cat_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0  # normalize 0-1
                X.append(arr)
                y.append(label_idx)
                loaded += 1
            except Exception as e:
                skipped += 1

        print(f"OK ({loaded} loaded)")

    if skipped > 0:
        print(f"  [WARN] {skipped} images could not be loaded and were skipped")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n[INFO] Dataset shape: X={X.shape}, y={y.shape}")
    print(f"[INFO] Categories used: {categories}")

    return X, y, categories


# ─── MODEL BUILDER ────────────────────────────────────────────────────────────
def build_model(num_classes):
    """
    Build a MobileNetV2-based transfer learning model.
    Falls back to a simple custom CNN if MobileNetV2 is unavailable.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.optimizers import Adam

        print("[INFO] Building MobileNetV2 transfer learning model...")

        # Load MobileNetV2 pretrained on ImageNet, without top layer
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet"
        )
        # Freeze base model weights — we only train the top layers
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax")
        ])

        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.summary()
        return model, "MobileNetV2"

    except Exception as e:
        print(f"[WARN] MobileNetV2 failed ({e}). Using custom CNN instead.")
        return build_custom_cnn(num_classes)


def build_custom_cnn(num_classes):
    """Simple custom CNN — fallback if MobileNetV2 unavailable."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam

    print("[INFO] Building custom CNN model...")

    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Classifier head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model, "CustomCNN"


# ─── DATA AUGMENTATION ────────────────────────────────────────────────────────
def augment_data(X_train, y_train):
    """
    Apply simple data augmentation to training set:
    flips, rotations, brightness — to improve generalization.
    """
    try:
        import tensorflow as tf
        augmented_X = []
        augmented_y = []

        for img, label in zip(X_train, y_train):
            augmented_X.append(img)
            augmented_y.append(label)

            # Horizontal flip
            augmented_X.append(np.fliplr(img))
            augmented_y.append(label)

            # Vertical flip (less common but adds variety)
            augmented_X.append(np.flipud(img))
            augmented_y.append(label)

            # Brightness adjustment
            bright = np.clip(img * np.random.uniform(0.7, 1.3), 0, 1)
            augmented_X.append(bright.astype(np.float32))
            augmented_y.append(label)

        return np.array(augmented_X, dtype=np.float32), np.array(augmented_y)

    except Exception as e:
        print(f"[WARN] Augmentation failed: {e}. Using original data.")
        return X_train, y_train


# ─── TRAIN ────────────────────────────────────────────────────────────────────
def train(use_demo=False):
    print("\n=== AI Civic Guardian - Image Classifier Training ===\n")

    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"[INFO] TensorFlow version: {tf.__version__}")
    except ImportError:
        print("[ERROR] TensorFlow not installed.")
        print("  Run: pip install tensorflow")
        sys.exit(1)

    from sklearn.model_selection import train_test_split

    # Generate demo images if requested
    if use_demo:
        generate_demo_images()

    # Check if images folder has data
    if not os.path.exists(IMAGES_DIR):
        print(f"[ERROR] Images directory not found: {IMAGES_DIR}")
        print("  Run with --demo flag to generate test images:")
        print("  python train_image_classifier.py --demo")
        sys.exit(1)

    # Load dataset
    X, y, categories = load_dataset()

    if len(X) == 0:
        print("\n[ERROR] No images loaded. Options:")
        print("  1. Add images to dataset/images/<category>/ folders")
        print("  2. Run: python train_image_classifier.py --demo")
        sys.exit(1)

    num_classes = len(categories)
    print(f"\n[INFO] Training with {num_classes} classes, {len(X)} total images")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    # Augment training data
    print("[INFO] Applying data augmentation...")
    X_train, y_train = augment_data(X_train, y_train)
    print(f"[INFO] After augmentation — Train: {len(X_train)}")

    # Build model
    model, model_type = build_model(num_classes)

    # Callbacks
    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    print(f"\n[INFO] Training for up to {EPOCHS} epochs (early stopping enabled)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULT] Test Accuracy: {test_acc * 100:.2f}%")
    print(f"[RESULT] Test Loss:     {test_loss:.4f}")

    # Detailed report
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    from sklearn.metrics import classification_report
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))

    # Save model
    model.save(MODEL_H5_PATH)
    print(f"\n[SAVED] Model saved to: {MODEL_H5_PATH}")

    # Save class names
    with open(CLASSES_PATH, "wb") as f:
        pickle.dump(categories, f)
    print(f"[SAVED] Class names saved to: {CLASSES_PATH}")

    # Save info file
    best_val_acc = max(history.history.get("val_accuracy", [test_acc]))
    info_text = f"""
AI Civic Guardian - Image Classifier Training Info
====================================================
Model Type   : {model_type}
Image Size   : {IMG_SIZE[0]}x{IMG_SIZE[1]}
Categories   : {categories}
Total Images : {len(X)}
Train Size   : {len(X_train)} (after augmentation)
Test Size    : {len(X_test)}
Epochs Run   : {len(history.history['accuracy'])}
Test Accuracy: {test_acc * 100:.2f}%
Best Val Acc : {best_val_acc * 100:.2f}%
Model File   : {MODEL_H5_PATH}
"""
    with open(INFO_PATH, "w") as f:
        f.write(info_text)
    print(f"[SAVED] Training info saved to: {INFO_PATH}")

    print("\n[DONE] Image classifier training complete!")

    # Quick prediction test
    print("\n--- Quick Prediction Test (first 3 test images) ---")
    for i in range(min(3, len(X_test))):
        img   = X_test[i:i+1]
        probs = model.predict(img, verbose=0)[0]
        pred_idx  = np.argmax(probs)
        pred_label = categories[pred_idx]
        confidence = probs[pred_idx] * 100
        actual     = categories[y_test[i]]
        print(f"  Actual: {actual:15s} → Predicted: {pred_label:15s} ({confidence:.1f}%)")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_mode = "--demo" in sys.argv
    train(use_demo=demo_mode)
