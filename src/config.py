import os
import pathlib
import torch

# --- Path Configurations ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Parameters ---
# Example parameters - adjust based on the actual model architecture
MODEL_NAME = "GazeGuardNet_v1"
INPUT_SHAPE = (3, 224, 224)  # Example: (channels, height, width)
NUM_CLASSES = 2  # Example: Focused vs. Distracted
EMBEDDING_DIM = 128
DROPOUT_RATE = 0.3

# --- Training Parameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
OPTIMIZER = "Adam"  # Options: "Adam", "SGD", etc.
LOSS_FUNCTION = "CrossEntropyLoss" # Options: "CrossEntropyLoss", "BCEWithLogitsLoss", etc.
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.1
VALIDATION_SPLIT = 0.2 # Percentage of training data to use for validation

# --- Environment Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1

# --- Data Preprocessing ---
IMAGE_MEAN = [0.485, 0.456, 0.406] # ImageNet mean
IMAGE_STD = [0.229, 0.224, 0.225] # ImageNet std
RESIZE_DIM = (INPUT_SHAPE[1], INPUT_SHAPE[2]) # (height, width)

# --- Logging ---
LOG_FILE = LOG_DIR / "gaze_guard_training.log"
TENSORBOARD_LOG_DIR = LOG_DIR / "tensorboard"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Checkpoint Saving ---
CHECKPOINT_FILENAME_TEMPLATE = f"{MODEL_NAME}_epoch_{{epoch:02d}}_val_loss_{{val_loss:.4f}}.pt"
BEST_MODEL_FILENAME = f"{MODEL_NAME}_best.pt"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Example of how to access configurations
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Input Shape: {INPUT_SHAPE}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Log File: {LOG_FILE}")
    print(f"Best Model Filename: {BEST_MODEL_FILENAME}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")