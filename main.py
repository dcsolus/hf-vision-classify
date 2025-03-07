import io
import time
import torch
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature
from utils.set_logger import MyLogger

logger = MyLogger()

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
logger.info(f"Using device: {device}")

# Load the CSV and convert image strings to binary
file_path = Path("resources", "test_02_ml.csv")
df = pd.read_csv(file_path)
logger.info(df.info())

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

logger.info(f"Processed {len(df)} images successfully")
logger.info(df.info())
logger.info(df.head())

# Encode the labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Convert numpy.int64 to regular Python int in the label mapping
label_mapping = {
    label: int(idx) 
    for label, idx in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
}
logger.info(f"Label mapping: {label_mapping}")

# Create id2label and label2id with Python integers
num_labels = len(label_mapping)
id2label = {int(id): label for label, id in label_mapping.items()}
label2id = {label: int(id) for id, label in id2label.items()}

logger.info(f"Label mapping: {label_mapping}")
logger.info(f"Number of classes: {num_labels}")

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

# Define a preprocessing function to handle binary image data.
def preprocess(example):
    # Convert binary data to a PIL image in RGB mode
    image = Image.open(io.BytesIO(example["image"])).convert("RGB")
    # Process the image using the image processor
    processed = image_processor(image, return_tensors="pt")
    # processed["pixel_values"] has shape [1, C, H, W]. Remove the batch dimension, enforce contiguous layout.
    example["pixel_values"] = processed["pixel_values"].squeeze(0).contiguous().to(device)
    return example

# Apply the preprocessing function to the dataset
dataset = dataset.map(preprocess)

# Set the format of the dataset to PyTorch tensors
dataset.set_format(type="torch", columns=["pixel_values", "label"])

# Split the dataset
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Move model to device after dataset preparation
model = model.to(device)

# Use a lower batch size for MPS devices
batch_size = 16

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    eval_strategy="epoch",  # evaluation_strategy="epoch" is deprecated but still works
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
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
    input_tensor = sample_input["pixel_values"].unsqueeze(0).to(device)
    
    # Get model prediction for signature
    with torch.no_grad():
        model.eval()
        sample_output = model(input_tensor)
    
    # Convert to numpy arrays for MLflow
    input_array = input_tensor.cpu().numpy()
    output_array = sample_output.logits.cpu().numpy()
    
    # Create signature
    signature = infer_signature(input_array, output_array)
    
    # Save the original forward method if needed
    original_forward = model.forward

    def patched_forward(x):
        outputs = original_forward(x)
        return outputs.logits

    # Override the forward method to return only logits
    model.forward = patched_forward

    # Log the model with MLflow using the model signature, without passing an input example
    mlflow.pytorch.log_model(
        model, 
        "model",
        signature=signature
    )
    
    # Prepare an input example as a JSON-compatible n-dimensional array (i.e. a list)
    input_example = input_array.tolist()
    
    # Generate a serving input example from the input example
    serving_input = mlflow.models.convert_input_example_to_serving_input(input_example)
    
    # Validate the model input using the serving input example
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    try:
        mlflow.models.validate_serving_input(model_uri, serving_input)
        logger.info("Model input validation succeeded")
    except Exception as e:
        logger.error(f"Model input validation failed: {e}", exc_info=True)