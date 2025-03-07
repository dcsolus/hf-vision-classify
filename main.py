import pandas as pd
from pathlib import Path

import io
import time
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature
import mlflow.pytorch

# Add device selection logic after the imports
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Get the appropriate device
device = get_device()
print(f"Using device: {device}")

# Load the CSV and convert image strings to binary
file_path = Path("resources", "test_02_ml.csv")
df = pd.read_csv(file_path)
print(df.info())

# Convert string representation of binary data back to binary
def hex_str_to_bytes(hex_str):
    # Remove the surrounding brackets and any extra whitespace
    hex_str = hex_str.strip('[]').replace(' ', '')
    # Convert the cleaned hex string to a bytes object
    return bytes.fromhex(hex_str)

# Apply the conversion to the image column
df['image'] = df['image'].apply(hex_str_to_bytes)

# Remove any rows where image conversion failed
df = df.dropna(subset=['image'])

print(f"Processed {len(df)} images successfully")
print(df.info())
print(df.head())

# Encode the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Convert numpy.int64 to regular Python int in the label mapping
label_mapping = {
    label: int(idx) 
    for label, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
}
print(f"Label mapping: {label_mapping}")

# Create id2label and label2id with Python integers
num_labels = len(label_mapping)
id2label = {int(id): label for label, id in label_mapping.items()}
label2id = {label: int(id) for id, label in id2label.items()}

print(f"Label mapping: {label_mapping}")
print(f"Number of classes: {num_labels}")

dataset = Dataset.from_pandas(df)

# Specify your pre-trained model
model_name = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(
    model_name, 
    use_fast=True,
    do_resize=True,
    size=224
)

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

# Initialize the classifier weights
model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
torch.nn.init.xavier_uniform_(model.classifier.weight)
model.classifier.bias.data.fill_(0)

model = model.to(device)

# Define a preprocessing function to handle binary image data.
def preprocess(example):
    # Convert binary data to a PIL image and ensure it's in RGB format.
    image = Image.open(io.BytesIO(example["image"])).convert("RGB")
    # Process the image using the image processor.
    processed = image_processor(image, return_tensors="pt")
    # Remove the batch dimension (which is 1) to get a tensor of shape [C, H, W].
    example["pixel_values"] = processed["pixel_values"].squeeze()
    example["pixel_values"] = example["pixel_values"].to(device)
    return example

# Apply the preprocessing function to the dataset.
dataset = dataset.map(preprocess)

# Set the format of the dataset to PyTorch tensors, specifying the columns to include.
dataset.set_format(type="torch", columns=["pixel_values", "label"])

# Split the dataset into training and evaluation subsets (e.g., an 80-20 split).
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Define improved training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,  # Increased from 8
    per_device_eval_batch_size=16,   # Increased from 8
    num_train_epochs=5,              # Increased from 3
    eval_strategy="epoch", # evaluation_strategy="epoch" is deprecated
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,             # Added learning rate
    weight_decay=0.01,              # Added weight decay
    warmup_ratio=0.1,               # Added warmup
    load_best_model_at_end=True,    # Load the best model when training ends
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
    report_to=['tensorboard'],
)

# Define a metric function for binary classification accuracy.
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Initialize MLflow for local tracking
experiment_name = "image_classification_experiment"
mlflow.set_tracking_uri("file:./mlruns")  # Store MLflow data locally
mlflow.set_experiment(experiment_name)

# Start an MLflow run with more detailed logging
run_name = f"vit-classification-{time.strftime('%Y%m%d-%H%M%S')}"
with mlflow.start_run(run_name=run_name):
    # Log system info and hyperparameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_classes", num_labels)
    mlflow.log_param("device", device)
    mlflow.log_params(training_args.to_dict())
    
    # Log label mapping
    mlflow.log_param("label_mapping", str(label_mapping))

    # Initialize and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and log metrics
    train_result = trainer.train()
    train_metrics = train_result.metrics
    
    # Log training metrics
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    
    # Evaluate and log metrics
    eval_metrics = trainer.evaluate()
    mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})

    # Get a sample input and prepare it for signature
    sample_input = next(iter(eval_dataset))
    input_tensor = sample_input["pixel_values"].unsqueeze(0)
    
    # Get model prediction for signature
    with torch.no_grad():
        model.eval()
        sample_output = model(input_tensor)
    
    # Convert to numpy arrays for MLflow
    input_array = input_tensor.cpu().numpy()
    output_array = sample_output.logits.cpu().numpy()
    
    # Create signature
    signature = infer_signature(input_array, output_array)

    # Save the model locally
    trainer.save_model(f"./model/{run_name}")
    
    # Log the model with MLflow
    mlflow.pytorch.log_model(
        model, 
        "model",
        signature=signature,
        input_example=input_array  # Pass the numpy array directly
    )

    # Validate the model input
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mlflow.models.validate_serving_input(model_uri, input_array)
