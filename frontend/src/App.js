import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
    const [formData, setFormData] = useState({
        "Dst Port": "",
        "Flow Duration": "",
        "Tot Fwd Pkts": "",
        "Tot Bwd Pkts": "",
        "Flow Byts/s": "",
        "Flow Pkts/s": "",
        "Pkt Len Max": "",
        "Pkt Len Mean": "",
        "Pkt Len Std": "",
        "SYN Flag Cnt": "",
        "ACK Flag Cnt": "",
        "FIN Flag Cnt": ""
    });

    const featureLabels = {
        "Dst Port": "Destination Port",
        "Flow Duration": "Flow Duration (ms)",
        "Tot Fwd Pkts": "Total Forward Packets",
        "Tot Bwd Pkts": "Total Backward Packets",
        "Flow Byts/s": "Flow Bytes per Second",
        "Flow Pkts/s": "Flow Packets per Second",
        "Pkt Len Max": "Packet Length Maximum",
        "Pkt Len Mean": "Packet Length Mean",
        "Pkt Len Std": "Packet Length Standard Deviation",
        "SYN Flag Cnt": "SYN Flag Count",
        "ACK Flag Cnt": "ACK Flag Count",
        "FIN Flag Cnt": "FIN Flag Count"
    };

    const [prediction, setPrediction] = useState("");
    const [reason, setReason] = useState("");

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            console.log("📌 Sending request to API:", formData);

            const response = await axios.post("http://127.0.0.1:5000/predict", formData);

            console.log("📌 API Response:", response.data);

            if (response.data && response.data.prediction) {
                setPrediction(response.data.prediction);
                
                // Extract full reason with feature name mapping
                let reasonText = response.data.reason || "No explanation provided.";
                Object.keys(featureLabels).forEach((key) => {
                    reasonText = reasonText.replace(key, featureLabels[key]);
                });

                setReason(reasonText);
            } else {
                setPrediction("Unexpected response format");
                setReason("");
            }
        } catch (error) {
            console.error("❌ API Error:", error);
            setPrediction("Error making prediction.");
            setReason("");
        }
    };

    return (
        <div className="container">
            <h1>Network Intrusion Detection</h1>
            <form onSubmit={handleSubmit}>
                {Object.keys(formData).map((feature, index) => (
                    <div key={index} className="input-group">
                        <label>{featureLabels[feature]}</label>
                        <input
                            type="number"
                            name={feature}
                            placeholder={featureLabels[feature]}
                            onChange={handleChange}
                            required
                        />
                    </div>
                ))}
                <button type="submit">Predict</button>
            </form>

            {prediction && (
                <div>
                    <h2>Prediction: {prediction}</h2>
                    {reason && (
                        <p className="reason">
                            <strong>Explanation: </strong> {reason}
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}

export default App;
