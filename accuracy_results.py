# accuracy_results.py

import preprocess
import npy
import load_and_process_json
import train_model
import model
import app
import chatbot
import matplotlib.pyplot as plt

def main():
    # Initialize results dictionary
    accuracy_results = {}

    # Preprocess.py accuracy check
    try:
        preprocess_accuracy = preprocess.evaluate_preprocessing()  # Ensure the function exists in preprocess.py
        accuracy_results["Preprocess"] = preprocess_accuracy
    except AttributeError:
        accuracy_results["Preprocess"] = None

    # Npy.py accuracy check
    try:
        npy_accuracy = npy.evaluate_npy_conversion()  # Ensure the function exists in npy.py
        accuracy_results["NPY"] = npy_accuracy
    except AttributeError:
        accuracy_results["NPY"] = None

    # Load_and_process_json.py accuracy check
    try:
        json_accuracy = load_and_process_json.evaluate_json_processing()  # Ensure the function exists
        accuracy_results["JSON Processing"] = json_accuracy
    except AttributeError:
        accuracy_results["JSON Processing"] = None

    # Train_model.py accuracy check
    try:
        train_accuracy = train_model.evaluate_model_training()  # Ensure the function exists
        accuracy_results["Model Training"] = train_accuracy
    except AttributeError:
        accuracy_results["Model Training"] = None

    # Model.py accuracy check
    try:
        model_accuracy = model.evaluate_model()  # Ensure the function exists
        accuracy_results["Model"] = model_accuracy
    except AttributeError:
        accuracy_results["Model"] = None

    # App.py accuracy check
    try:
        app_accuracy = app.evaluate_application()  # Ensure the function exists
        accuracy_results["App"] = app_accuracy
    except AttributeError:
        accuracy_results["App"] = None

    # Chatbot.py accuracy check
    try:
        chatbot_accuracy = chatbot.evaluate_chatbot()  # Ensure the function exists
        accuracy_results["Chatbot"] = chatbot_accuracy
    except AttributeError:
        accuracy_results["Chatbot"] = None

    # Save text-based results to a .txt file
    save_text_results(accuracy_results)

    # Generate graph and save as PNG/JPEG
    save_graph(accuracy_results)

def save_text_results(results):
    with open("accuracy_results.txt", "w") as file:
        for module, accuracy in results.items():
            if accuracy is not None:
                file.write(f"{module}: {accuracy:.2f}%\n")
            else:
                file.write(f"{module}: Accuracy function not implemented.\n")
    print("Text-based results saved to accuracy_results.txt")

def save_graph(results):
    modules = list(results.keys())
    accuracies = [accuracy if accuracy is not None else 0 for accuracy in results.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(modules, accuracies, color="skyblue")
    plt.title("Accuracy Results by Module", fontsize=16)
    plt.xlabel("Modules", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the graph as PNG and JPEG
    plt.savefig("accuracy_results.png", format="png")
    plt.savefig("accuracy_results.jpeg", format="jpeg")
    print("Graph saved as accuracy_results.png and accuracy_results.jpeg")
    plt.close()

if __name__ == "__main__":
    main()
