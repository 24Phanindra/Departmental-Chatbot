# accuracy_results.py

import preprocess
import npy
import load_and_process_json
import train_model
import model
import app
import chatbot

def main():
    # Initialize results dictionary
    accuracy_results = {}

    # Preprocess.py accuracy check
    try:
        preprocess_accuracy = preprocess.evaluate_preprocessing()  # Ensure the function exists in preprocess.py
        accuracy_results["Preprocess"] = preprocess_accuracy
    except AttributeError:
        accuracy_results["Preprocess"] = "No accuracy function found in preprocess.py"

    # Npy.py accuracy check
    try:
        npy_accuracy = npy.evaluate_npy_conversion()  # Ensure the function exists in npy.py
        accuracy_results["NPY"] = npy_accuracy
    except AttributeError:
        accuracy_results["NPY"] = "No accuracy function found in npy.py"

    # Load_and_process_json.py accuracy check
    try:
        json_accuracy = load_and_process_json.evaluate_json_processing()  # Ensure the function exists
        accuracy_results["JSON Processing"] = json_accuracy
    except AttributeError:
        accuracy_results["JSON Processing"] = "No accuracy function found in load_and_process_json.py"

    # Train_model.py accuracy check
    try:
        train_accuracy = train_model.evaluate_model_training()  # Ensure the function exists
        accuracy_results["Model Training"] = train_accuracy
    except AttributeError:
        accuracy_results["Model Training"] = "No accuracy function found in train_model.py"

    # Model.py accuracy check
    try:
        model_accuracy = model.evaluate_model()  # Ensure the function exists
        accuracy_results["Model"] = model_accuracy
    except AttributeError:
        accuracy_results["Model"] = "No accuracy function found in model.py"

    # App.py accuracy check
    try:
        app_accuracy = app.evaluate_application()  # Ensure the function exists
        accuracy_results["App"] = app_accuracy
    except AttributeError:
        accuracy_results["App"] = "No accuracy function found in app.py"

    # Chatbot.py accuracy check
    try:
        chatbot_accuracy = chatbot.evaluate_chatbot()  # Ensure the function exists
        accuracy_results["Chatbot"] = chatbot_accuracy
    except AttributeError:
        accuracy_results["Chatbot"] = "No accuracy function found in chatbot.py"

    # Print results
    print("Accuracy Results:")
    for module, accuracy in accuracy_results.items():
        print(f"{module}: {accuracy}")

if __name__ == "__main__":
    main()
