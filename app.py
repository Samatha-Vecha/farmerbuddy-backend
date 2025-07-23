from flask import Flask, request, jsonify
import pickle
import joblib
import pandas as pd
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, auth, firestore
import os
import numpy as np
from google.cloud.firestore_v1 import ArrayUnion

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Load Firebase credentials
FIREBASE_CREDENTIALS = "firebase_config.json"
db = None
if os.path.exists(FIREBASE_CREDENTIALS):
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
else:
    print("⚠ Warning: Firebase configuration file missing! Authentication may not work.")

# Load ML Models & Transformers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for models
FERTILIZER_MODEL_PATH = os.path.join(BASE_DIR, "model/decision_tree_model.pkl")
FERTILIZER_TRANSFORMER_PATH = os.path.join(BASE_DIR, "model/transformer.pkl")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CROP_SCALER_PATH = os.path.join(BASE_DIR, "minmaxscaler.pkl")
CROP_DICT_PATH = os.path.join(BASE_DIR, "model", "crop_dict.pkl")

# Load Fertilizer Recommendation Model
fertilizer_model = None
fertilizer_transformer = None
if os.path.exists(FERTILIZER_MODEL_PATH) and os.path.exists(FERTILIZER_TRANSFORMER_PATH):
    with open(FERTILIZER_MODEL_PATH, "rb") as model_file:
        fertilizer_model = pickle.load(model_file)
    with open(FERTILIZER_TRANSFORMER_PATH, "rb") as transformer_file:
        fertilizer_transformer = pickle.load(transformer_file)
    print("✅ Fertilizer model and transformer loaded successfully!")
else:
    print("⚠ Warning: Fertilizer model or transformer file missing!")

# Load Crop Recommendation Model
try:
    crop_recommendation_model = pickle.load(open(CROP_MODEL_PATH, 'rb'))
    crop_scaler = pickle.load(open(CROP_SCALER_PATH, 'rb'))  # used for input normalization
    print("✅ Crop recommendation model and scaler loaded successfully!")
except Exception as e:
    crop_recommendation_model = None
    crop_scaler = None
    print("⚠ Error loading crop recommendation model or scaler:", e)

# Load Crop Label Dictionary

crop_dict = {}
reverse_crop_dict = {}
if os.path.exists(CROP_DICT_PATH):
    with open(CROP_DICT_PATH, "rb") as f:
        raw_dict = pickle.load(f)
        # ✅ No conversion to int — just reverse it safely
        crop_dict = {v: k for k, v in raw_dict.items()}  # int → crop name
        reverse_crop_dict = raw_dict                    # crop name → int
    print("✅ Crop label dictionary loaded!")
else:
    print("⚠ Warning: crop_dict.pkl not found!")


# User Registration Route
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        email = data.get("email", "")
        password = data.get("password", "")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        user = auth.create_user(email=email, password=password)
        return jsonify({"message": "User registered successfully", "uid": user.uid}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# User Login Route
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        email = data.get("email", "")
        password = data.get("password", "")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        user = auth.get_user_by_email(email)
        return jsonify({"message": "Login successful", "uid": user.uid})

    except firebase_admin.auth.UserNotFoundError:
        return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Fertilizer Prediction Route
@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    if fertilizer_model is None or fertilizer_transformer is None:
        return jsonify({"error": "Fertilizer model not loaded!"}), 500

    data = request.get_json(silent=True)
    if not data or "uid" not in data or "input" not in data:
        return jsonify({"error": "Invalid request. UID and input data required!"}), 400

    uid = data["uid"]
    input_data = pd.DataFrame([data["input"]])
    try:
        transformed_input = fertilizer_transformer.transform(input_data)
        prediction = fertilizer_model.predict(transformed_input)[0]

        if db:
            db.collection("fertilizer_predictions").document(uid).set({
                "history": ArrayUnion([{
                    "input": data["input"],
                    "prediction": prediction
                }])
            }, merge=True)

        return jsonify({"fertilizer": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Crop Name Prediction Route
@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    if crop_recommendation_model is None or crop_scaler is None:
        return jsonify({"error": "Crop recommendation model not loaded!"}), 500

    data = request.get_json(silent=True)
    if not data or "uid" not in data or "input" not in data:
        return jsonify({"error": "Invalid request. UID and input data required!"}), 400

    uid = data["uid"]
    input_data = data["input"]

    try:
        # Prepare input dataframe
        df_input = pd.DataFrame([input_data])

        # Apply scaling
        scaled_input = crop_scaler.transform(df_input)

        # Make prediction
        prediction_index = int(crop_recommendation_model.predict(scaled_input)[0])
        predicted_crop = crop_dict.get(prediction_index, "Unknown")

        # Optional: store in Firebase
        if db:
            db.collection("crop_recommendation_predictions").document(uid).set({
            "history": ArrayUnion([{
                "input": {k: (v.item() if isinstance(v, np.generic) else v) for k, v in input_data.items()},
                "predicted_crop": predicted_crop
            }])
        }, merge=True)


        return jsonify({"predicted_crop": predicted_crop})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# History Route
@app.route("/history/<uid>", methods=["GET"])
def history(uid):
    if db is None:
        return jsonify({"error": "Database not connected!"}), 500

    try:
        fertilizer_doc = db.collection("fertilizer_predictions").document(uid).get()
        crop_doc = db.collection("crop_recommendation_predictions").document(uid).get()

        history_data = {
            "fertilizer_history": fertilizer_doc.to_dict().get("history", []) if fertilizer_doc.exists else [],
            "crop_history": crop_doc.to_dict().get("history", []) if crop_doc.exists else []
        }


        return jsonify(history_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Delete Route
@app.route("/delete/<collection>/<doc_id>", methods=["DELETE"])
def delete_prediction(collection, doc_id):
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        collection_name = "fertilizer_predictions" if collection == "fertilizer_predictions" else "crop_name"
        doc_ref = db.collection(collection_name).document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({"error": f"Document with ID {doc_id} not found in {collection_name}"}), 404

        doc_ref.delete()
        return jsonify({"success": True, "message": f"Prediction {doc_id} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Error deleting prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
