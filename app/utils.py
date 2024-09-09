import pickle

def get_model(file_path):
    """Load the saved model from the specified file path."""
    with open(file_path, "rb") as model_file:
        return pickle.load(model_file)

def get_scaler(file_path):
    """Load the saved scaler from the specified file path."""
    with open(file_path, "rb") as scaler_file:
        return pickle.load(scaler_file)


def get_prediction(model, scaler, features: list):
    """Make predictions using the loaded model and scaler."""
    prediction = model.predict(scaler.transform([features]))
    return "Diabetic" if prediction == 1 else "Non-Diabetic"
