import json

from keras.models import model_from_json


def save_model(model, architecture_fp, weights_fp):
    """
    Save architecture and weights for a Keras model
    Args:
        model (keras model): A Keras model
        architecture_fp (str): File path where to save the model
        architecture
        (in JSON format)
        weights_fp (str): File path where to save the model weights
        (in HDF5 format)
    """
    # Save weights
    model.save_weights(weights_fp)
    # Save architecture
    saved_model = model.to_json()
    with open(architecture_fp, 'w') as architecture_file:
        json.dump(saved_model, architecture_file)


def load_model(architecture_fp, weights_fp):
    """
    Load architecture and weights for a Keras model
    Args:
        architecture_fp (str): File path to the model architecture
        (in JSON format)
        weights_fp (str): File path to the model weights (in HDF5 format)
    Retruns:
        A Keras trained model
    """
    # Load architecture
    with open(architecture_fp, 'r') as architecture_file:
        model_architecture = json.load(architecture_file)

    loaded_model = model_from_json(model_architecture)

    # Load weights
    loaded_model.load_weights(weights_fp)
    return loaded_model
