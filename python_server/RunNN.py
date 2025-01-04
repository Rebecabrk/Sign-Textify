import onnxruntime
import numpy as np

def load_model(model_path):
    """Loads an ONNX model."""
    return onnxruntime.InferenceSession(model_path)

def normalize_landmarks(landmarks):
    """Normalizes landmarks to a flat array suitable for the model."""
    flat_landmarks = [coord for point in landmarks for coord in point]
    return np.array(flat_landmarks, dtype=np.float32).reshape(1, -1)

def predict_label(landmarks, session):
    """Runs inference on the normalized landmarks and returns the predicted label."""
    normalized_landmarks = normalize_landmarks(landmarks)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: normalized_landmarks})
    return np.argmax(result[0])  # Assuming the output is a softmax array