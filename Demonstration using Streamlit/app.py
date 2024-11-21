import streamlit as st
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from final_layer_retraining import train_and_evaluate_final_layer_retraining

# Define available models
models = [
    "Weight Distillation",
    "Final Layer Retraining",
    "Transfer Learning",
    "Learn++",
    "CatBoost",
    "Elastic Weight Consolidation (Regularization Based)"
]

# Title and description
st.title("Incremental Learning Demonstration")

st.write("Future Work: Showcasing the Catastrophic Forgetting on page 1, and the incremental learning demonstration on page 2.")
st.write("Explain the cases of having having access to new data and not having access to new data.\n")
st.write("If yes, then Final Layer Retraining, Weight Distillation, Transfer Learning being suitable\n")
st.write("If not, then Learn++, CatBoost, Elastic Weight Consolidation being suitable\n")

st.write("Select a dataset and a model to visualize incremental learning.")

# Dataset selection
dataset_name = st.selectbox("Select Dataset", ["MNIST", "CIFAR-10"])

# Load datasets


@st.cache_data
def load_data(dataset_name):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = datasets.MNIST(
            root="data/MNIST", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root="data/MNIST", train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR-10":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.CIFAR10(
            root="data/CIFAR10", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            root="data/CIFAR10", train=False, download=True, transform=transform)
    return train_data, test_data


train_data, test_data = load_data(dataset_name)


# Dataset information
dataset_info = {
    "MNIST": {
        "description": (
            "The MNIST dataset is a collection of 70,000 handwritten digits (0-9) commonly used for training "
            "various image processing systems. It contains 60,000 training images and 10,000 testing images, "
            "each of size 28x28 pixels. Given the dataset's size and diversity, you may have sufficient samples "
            "for fine-tuning, but be mindful of data augmentation techniques to improve model robustness against new styles."
        )
    },
    "CIFAR-10": {
        "description": (
            "CIFAR-10 consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. "
            "The classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset "
            "is divided into 50,000 training images and 10,000 testing images, which is standard for training and evaluating "
            "machine learning models. Each class is distinct, ensuring clear classification tasks. Due to its balanced distribution "
            "and moderate size, CIFAR-10 is popular in academic and research settings for benchmarking algorithms in image recognition "
            "and computer vision."
        )
    }
}

# Display dataset details with a checkbox
if st.checkbox("Show Dataset Details"):
    st.write(f"### {dataset_name} Dataset Details")
    st.write(dataset_info[dataset_name]["description"])
    st.write(f"Number of training samples: {len(train_data)}")
    st.write(f"Number of testing samples: {len(test_data)}")

    # Display a few sample images from the selected dataset
    st.write("Sample Images:")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i, ax in enumerate(axes.flat):
        image, label = train_data[i]
        ax.imshow(image.permute(1, 2, 0).squeeze(),
                  cmap="gray" if dataset_name == "MNIST" else None)
        ax.set_title(f"Label: {label}")
        ax.axis("off")
    st.pyplot(fig)


# Model selection
model_name = st.selectbox("Select Model", models)

if model_name == "Final Layer Retraining":
    st.write("### Final Layer Retraining Model Selected")

    # Train and evaluate the model
    results = train_and_evaluate_final_layer_retraining(
        train_data, test_data, dataset_name)

    # Display all results
    st.write("## Experiment Time Taken")
    st.write(
        f"Initial model training time: {results['initial_train_time']:.2f} seconds")
    st.write(
        f"Final model training time: {results['final_train_time']:.2f} seconds")

    st.write("## Initial Model Summary")
    st.text(results["initial_model_summary"])

    st.write("## Initial Model Training Epochs")
    for epoch_log in results["initial_training_epochs"]:
        st.write(
            f"Epoch {epoch_log['epoch']}: Loss={epoch_log['loss']:.4f}, Accuracy={epoch_log['accuracy']:.4f}")

    st.write("## Initial Model Evaluation (0-7 Test Data)")
    st.write(
        f"Loss: {results['initial_evaluation_0_7']['loss']:.4f}, Accuracy: {results['initial_evaluation_0_7']['accuracy']:.4f}")

    st.write("## Final Model Summary")
    st.text(results["final_model_summary"])

    st.write("## Final Model Training Epochs")
    for epoch_log in results["final_training_epochs"]:
        st.write(
            f"Epoch {epoch_log['epoch']}: Loss={epoch_log['loss']:.4f}, Accuracy={epoch_log['accuracy']:.4f}")

    st.write("## Final Model Evaluations")
    st.write("Evaluation on 0-7 test set:")
    st.write(
        f"Loss: {results['final_evaluation_0_7']['loss']:.4f}, Accuracy: {results['final_evaluation_0_7']['accuracy']:.4f}")

    st.write("Evaluation on 8-9 test set:")
    st.write(
        f"Loss: {results['final_evaluation_8_9']['loss']:.4f}, Accuracy: {results['final_evaluation_8_9']['accuracy']:.4f}")

    st.write("Evaluation on full test set:")
    st.write(
        f"Loss: {results['final_evaluation_full']['loss']:.4f}, Accuracy: {results['final_evaluation_full']['accuracy']:.4f}")
