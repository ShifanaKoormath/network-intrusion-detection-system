from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import shap
import numpy as np
from sklearn.preprocessing import StandardScaler

# ✅ Load model, scaler, and sample training data
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
X_TRAIN_PATH = os.path.join(os.path.dirname(__file__), "X_train_sample.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(X_TRAIN_PATH):
    raise FileNotFoundError("❌ Model, scaler, or X_train_sample file is missing!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_train_sample = joblib.load(X_TRAIN_PATH)

# ✅ Define important features
important_features = [
    "Dst Port", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s", "Pkt Len Max", "Pkt Len Mean",
    "Pkt Len Std", "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt"
]

# ✅ SHAP Explainer
shap_explainer = shap.Explainer(model, X_train_sample)

# ✅ Feature explanations
feature_explanations = {
    "Dst Port": "Unusual destination port activity detected, which may indicate unauthorized access attempts.",
    "Flow Duration": "Long flow duration may suggest slow data exfiltration or prolonged attack attempts.",
    "Tot Fwd Pkts": "High number of forward packets can indicate an attempted brute-force attack.",
    "Tot Bwd Pkts": "Large number of backward packets suggests abnormal response behavior.",
    "Flow Byts/s": "Unusually high or low byte flow rate might indicate scanning or data leakage.",
    "Flow Pkts/s": "Irregular packet per second rate may indicate a DoS attack attempt.",
    "Pkt Len Max": "Maximum packet length being too high or low can signal evasion techniques.",
    "Pkt Len Mean": "Mean packet length deviation from the norm may indicate suspicious activity.",
    "Pkt Len Std": "High variation in packet lengths suggests mixed traffic, possibly an attack attempt.",
    "SYN Flag Cnt": "Frequent SYN flag usage may indicate SYN flood attacks or excessive connection attempts.",
    "ACK Flag Cnt": "High ACK flag count suggests response manipulation, potentially avoiding detection.",
    "FIN Flag Cnt": "Multiple FIN flag occurrences indicate possible connection hijacking or stealth scanning."
}

# ✅ Flask App
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "🚀 Network Intrusion Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔹 Get JSON data
        data = request.get_json()

        # 🔹 Validate input
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected JSON object."}), 400

        # 🔹 Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # 🔹 Ensure required columns exist
        missing_features = [feat for feat in important_features if feat not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        df = df[important_features].astype(float)  # Convert to float

        # ✅ Check input shape
        print("📌 Input Shape:", df.shape)

        # 🔹 Scale features
        df_scaled = scaler.transform(df)

        # ✅ Check scaled shape
        print("📌 Scaled Input Shape:", df_scaled.shape)

        # 🔹 Predict
        prediction = model.predict(df_scaled)[0]
        result = "Malicious" if prediction == 1 else "Normal"

        if result == "Malicious":
            # ✅ SHAP Explanation
            shap_values = shap_explainer(df_scaled)

            # ✅ Ensure SHAP values have the correct shape
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Extract values if wrapped in a list

            if hasattr(shap_values, "values") and shap_values.values is not None:
                feature_importance = np.abs(shap_values.values).mean(axis=0)
            else:
                raise ValueError("❌ SHAP values are empty or not computed correctly.")

            # ✅ Debugging SHAP output
            print("📌 SHAP Values Shape:", feature_importance.shape)

            # ✅ Get Top 2 Influential Features
            top_feature_indices = np.argsort(feature_importance)[-2:][::-1]  # Get top 2
            top_features = [important_features[i] for i in top_feature_indices if i < len(important_features)]

            # ✅ Ensure we have at least 1 feature
            if len(top_features) == 0:
                explanation = "No significant features identified."
            elif len(top_features) == 1:
                explanation = f"{feature_explanations.get(top_features[0], 'Unknown reason')}."
            else:
                explanation = f"{feature_explanations.get(top_features[0], 'Unknown reason')}. Additionally, {feature_explanations.get(top_features[1], 'Unknown reason')}."

            # ✅ Log Response
            response = {"prediction": result, "reason": explanation}
        else:
            response = {"prediction": result}  # No explanation for normal traffic

        print("📌 API Response:", response)
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

    except Exception as e:
        print("❌ API Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
