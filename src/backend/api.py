import sys
import os
import torch
import pickle
import pandas as pd
import numpy as np
import torch.serialization
from flask import Flask, request, jsonify, Response
import subprocess
from flask_cors import CORS


# Add the '../' directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

app = Flask(__name__)
CORS(app)

# Define model paths
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"

torch.serialization.add_safe_globals([
    np.dtype,
    np.float64,
    np.dtypes.Float64DType,
    np.core.multiarray.scalar
])

device = torch.device("cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Import the model class
from fcu_model_LSTM import LSTMAutoencoder, preprocess_data, load_data, sample_data, train_model_kfold, test_model

# Load model
def load_model():
    input_dim = 16  # Ensure this matches your feature count
    model = LSTMAutoencoder(input_dim=input_dim, sequence_length=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['threshold']

# Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, best_threshold = load_model()

        # Get data from request
        data = request.json
        print("Received input data:", data)

        # Create dataframe from input
        df = pd.DataFrame([data])

        # Ensure all required columns exist
        if "Lab 409 Room Temp Deviation From Setpoint (°F)" not in df.columns:
            df["Lab 409 Room Temp Deviation From Setpoint (°F)"] = 0.0

        # Preprocess data
        processed_data, _ = preprocess_data(df, scaler=scaler, fit_scaler=False)

        # Convert to tensor
        tensor_data = torch.tensor(processed_data.values, dtype=torch.float32).unsqueeze(0).to(device)

        # Run model prediction
        with torch.no_grad():
            output = model(tensor_data)
            loss = torch.nn.functional.mse_loss(output, tensor_data).item()

        # Determine fault status
        prediction = 1 if loss > best_threshold else 0

        response = {
            "result": "Fault" if prediction == 1 else "Normal",
            "fault_probability": min(loss / (best_threshold * 2), 1.0),
            "mse_loss": loss,
            "threshold": best_threshold
        }

        print("Prediction result:", response)
        return jsonify(response)
    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/run-model', methods=['POST'])
def run_model():
    try:
        # Swap out these paths if needed
        TRAIN_FILE = "../fcu_training_data.csv"
        TEST_FILES = ["../test_data_f_labeled.csv", "../test_data_n_labeled.csv"]
        OUTPUT_CSV = "test_predictions_output.csv"

        train_df = load_data(TRAIN_FILE)
        train_df = sample_data(train_df)
        train_data, scaler = preprocess_data(train_df, fit_scaler=True)

        model, threshold = train_model_kfold(train_data, MODEL_PATH, scaler, k=5)

        # Save updated scaler
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        test_model(MODEL_PATH, TEST_FILES, OUTPUT_CSV)

        return jsonify({"status": "success", "message": "Model trained, and results saved."})

    except Exception as e:
        print("Error in /run-model:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/run-test', methods=['POST'])
def run_test():
    try:
        # NEW: Get which test files to use from frontend
        request_data = request.get_json()
        test_type = request_data.get("test_type", "both")

        if test_type == "fault":
            TEST_FILES = ["../test_data_f_labeled.csv"]
        elif test_type == "normal":
            TEST_FILES = ["../test_data_n_labeled.csv"]
        elif test_type == "mixed":
            TEST_FILES = ["../test_data_mixed.csv"]
        elif test_type == "mixedNoGaps":
            TEST_FILES = ["../test_data_mixedG.csv"]
        else:
            TEST_FILES = ["../test_data_f_labeled.csv", "../test_data_n_labeled.csv"]



        OUTPUT_CSV = "test_predictions_output.csv"

        # Capture printed output from test_model
        import io
        import contextlib
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            test_model(MODEL_PATH, TEST_FILES, OUTPUT_CSV)

        test_output = captured_output.getvalue()

        return jsonify({
            "status": "success",
            "message": "Test completed successfully.",
            "results": test_output,
            "test_type": test_type,
            "test_files": TEST_FILES
        })

    except Exception as e:
        print("Error in /run-test:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# THIS IS THE CORRECT WAY TO START THE SERVER
if __name__ == "__main__":
    print("Starting server on port 5000...")
    app.run(debug=True, port=5000)
