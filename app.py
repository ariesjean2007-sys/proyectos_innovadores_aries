"""
Pneumonia Detector - Servidor Flask con MobileNetV2
=====================================================
Clases (orden alfabético Keras):
  0: Bacterial Pneumonia
  1: Healthy
  2: Viral Pneumonia
  3: covid-19

USO:
  1. Borrar modelo viejo:  del pneumonia_model.h5
  2. Entrenar:  python app.py --train --data "Training Data-20260526T145220Z-3-001" --epochs 60
  3. Servidor:  python app.py
  4. Abrir:     index.html en el navegador
"""

import os, io, base64, argparse
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Constantes — mismo orden alfabético que usa Keras internamente ─────────────
IMG_SIZE   = 224
CLASSES    = ["Bacterial Pneumonia", "Healthy", "Viral Pneumonia", "covid-19"]
MODEL_PATH = "pneumonia_model.h5"

app = Flask(__name__)
CORS(app)
model = None

# ── Modelo ────────────────────────────────────────────────────────────────────
def build_model():
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, weights="imagenet"
    )
    base.trainable = False  # congelado al inicio

    x   = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dense(256, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.5)(x)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    m = tf.keras.Model(base.input, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return m

def load_model():
    global model
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        print(f"[+] Cargando modelo guardado: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("[!] No hay modelo entrenado. Ejecuta: python app.py --train --data <ruta>")
        model = build_model()

# ── Entrenamiento en 2 fases ──────────────────────────────────────────────────
def train(data_dir, epochs=60, batch_size=16):
    import tensorflow as tf

    print(f"\n[*] Buscando imágenes en: {os.path.abspath(data_dir)}")
    if os.path.isdir(data_dir):
        print(f"    Carpetas encontradas: {os.listdir(data_dir)}")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=0.15,
        rotation_range=20, width_shift_range=0.15,
        height_shift_range=0.15, zoom_range=0.2,
        horizontal_flip=True, brightness_range=[0.8, 1.2],
        shear_range=0.1, fill_mode="nearest"
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=0.15
    )
    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size,
        class_mode="categorical", subset="training", shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        data_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size,
        class_mode="categorical", subset="validation", shuffle=False
    )

    print(f"\n    Orden de clases (Keras): {train_gen.class_indices}")
    print(f"    Entrenamiento : {train_gen.samples} imgs  |  Validación: {val_gen.samples} imgs\n")

    # ── Fase 1: base congelada, 20 épocas ─────────────────────────────────────
    print("=" * 55)
    print(" FASE 1: Entrenando capas superiores (base congelada)")
    print("=" * 55)
    m = build_model()
    m.fit(train_gen, validation_data=val_gen, epochs=20,
          callbacks=[
              tf.keras.callbacks.ModelCheckpoint(
                  MODEL_PATH, save_best_only=True,
                  monitor="val_accuracy", verbose=1)
          ])

    # ── Fase 2: descongelar últimas 30 capas con lr muy bajo ──────────────────
    print("\n" + "=" * 55)
    print(" FASE 2: Fine-tuning (descongelando últimas 30 capas)")
    print("=" * 55)

    # Reconstruir con fine-tune activado
    base2 = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False, weights="imagenet"
    )
    base2.trainable = True
    for layer in base2.layers[:-30]:
        layer.trainable = False

    x2   = tf.keras.layers.GlobalAveragePooling2D()(base2.output)
    x2   = tf.keras.layers.BatchNormalization()(x2)
    x2   = tf.keras.layers.Dense(256, activation="relu")(x2)
    x2   = tf.keras.layers.Dropout(0.5)(x2)
    x2   = tf.keras.layers.Dense(128, activation="relu")(x2)
    x2   = tf.keras.layers.Dropout(0.3)(x2)
    out2 = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x2)
    m2   = tf.keras.Model(base2.input, out2)

    # Cargar pesos aprendidos en fase 1
    m2.load_weights(MODEL_PATH)

    m2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy", metrics=["accuracy"]
    )
    m2.fit(train_gen, validation_data=val_gen, epochs=epochs - 20,
           callbacks=[
               tf.keras.callbacks.ModelCheckpoint(
                   MODEL_PATH, save_best_only=True,
                   monitor="val_accuracy", verbose=1),
               tf.keras.callbacks.ReduceLROnPlateau(
                   monitor="val_loss", factor=0.5,
                   patience=5, min_lr=1e-7, verbose=1),
               tf.keras.callbacks.EarlyStopping(
                   monitor="val_accuracy", patience=12,
                   restore_best_weights=True, verbose=1)
           ])

    print(f"\n[✓] Modelo guardado: {MODEL_PATH}")

# ── Preprocesamiento ──────────────────────────────────────────────────────────
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    if request.files.get("image"):
        img_bytes = request.files["image"].read()
    elif request.json and request.json.get("image"):
        img_bytes = base64.b64decode(request.json["image"])
    else:
        return jsonify({"error": "No se recibió imagen"}), 400
    try:
        x     = preprocess(img_bytes)
        preds = model.predict(x, verbose=0)[0]
        idx   = int(np.argmax(preds))
        return jsonify({
            "class":      CLASSES[idx],
            "confidence": round(float(preds[idx]) * 100, 2),
            "all":        {c: round(float(p) * 100, 2) for c, p in zip(CLASSES, preds)}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    return jsonify({
        "model_loaded":  model is not None,
        "model_trained": os.path.exists(MODEL_PATH),
        "classes":       CLASSES
    })

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  action="store_true")
    parser.add_argument("--data",   default="./dataset")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    if args.train:
        train(args.data, epochs=args.epochs)
    else:
        load_model()
        print(f"\n[✓] Servidor corriendo en http://localhost:{args.port}")
        print("    Abre index.html en el navegador\n")
        app.run(port=args.port, debug=False)