import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_DATASET_PATH = 'gaze_dataset' # Example path structure: gaze_dataset/eyes_visible/*.jpg, gaze_dataset/eyes_away/*.jpg
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15 # Relative to the original dataset size, so effectively val_size / (1 - test_size) of the training set
RANDOM_STATE = 42
LABEL_MAP = {'eyes_visible': 1, 'eyes_away': 0}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Optional[np.ndarray]:
    """
    Preprocesses a single image: resize and convert to grayscale.

    Args:
        image: Input image as a NumPy array (BGR format from cv2).
        target_size: The target dimensions (width, height) for resizing.

    Returns:
        Preprocessed image as a NumPy array (grayscale) or None if preprocessing fails.
    """
    try:
        if image is None:
            logging.warning("Input image is None.")
            return None
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize image
        resized_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_AREA)
        # Normalize pixel values (optional, depending on the model)
        # normalized_image = resized_image / 255.0
        # return normalized_image.astype(np.float32)
        return resized_image # Return as uint8 for now
    except cv2.error as e:
        logging.error(f"OpenCV error during preprocessing: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during preprocessing: {e}")
        return None


def load_images_from_folder(
    folder_path: str,
    label: int,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Loads and preprocesses all images from a specific folder.

    Args:
        folder_path: Path to the folder containing images for a single class.
        label: The integer label corresponding to the class.
        target_size: The target dimensions for resizing images.

    Returns:
        A tuple containing:
        - A list of preprocessed image arrays.
        - A list of corresponding integer labels.
    """
    images = []
    labels = []
    if not os.path.isdir(folder_path):
        logging.warning(f"Folder not found: {folder_path}")
        return images, labels

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    logging.info(f"Loading images from: {folder_path} with label {label}")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in valid_extensions:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    logging.warning(f"Could not read image: {file_path}")
                    continue

                processed_image = preprocess_image(image, target_size)
                if processed_image is not None:
                    images.append(processed_image)
                    labels.append(label)
                else:
                    logging.warning(f"Failed to preprocess image: {file_path}")

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Loaded {len(images)} images from {folder_path}")
    return images, labels


def load_gaze_data(
    dataset_base_path: str = DEFAULT_DATASET_PATH,
    label_map: Dict[str, int] = LABEL_MAP,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads the complete gaze dataset from the specified base path.

    Assumes a directory structure like:
    dataset_base_path/
        class_name_1/
            image1.jpg
            image2.png
            ...
        class_name_2/
            image3.jpeg
            ...

    Args:
        dataset_base_path: The root directory of the dataset.
        label_map: A dictionary mapping class folder names to integer labels.
        target_size: The target dimensions for resizing images.

    Returns:
        A tuple containing:
        - A NumPy array of all preprocessed images (or None if loading fails).
        - A NumPy array of corresponding integer labels (or None if loading fails).
    """
    all_images = []
    all_labels = []
    found_data = False

    if not os.path.isdir(dataset_base_path):
        logging.error(f"Dataset base path not found: {dataset_base_path}")
        return None, None

    for class_name, label in label_map.items():
        class_folder_path = os.path.join(dataset_base_path, class_name)
        images, labels = load_images_from_folder(class_folder_path, label, target_size)
        if images:
            all_images.extend(images)
            all_labels.extend(labels)
            found_data = True

    if not found_data:
        logging.error(f"No images loaded from dataset path: {dataset_base_path}")
        return None, None

    # Convert lists to NumPy arrays
    # Add channel dimension for grayscale images if needed by the model (e.g., for CNNs)
    # Shape will be (num_samples, height, width, 1)
    try:
        X = np.array(all_images, dtype=np.uint8)
        y = np.array(all_labels, dtype=np.int32)

        # Reshape grayscale images to include channel dimension
        if len(X.shape) == 3: # (num_samples, height, width)
             X = np.expand_dims(X, axis=-1) # (num_samples, height, width, 1)

        # Shuffle the data
        indices = np.arange(X.shape[0])
        np.random.seed(RANDOM_STATE)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        logging.info(f"Dataset loaded successfully. Shape: X={X.shape}, y={y.shape}")
        return X, y

    except Exception as e:
        logging.error(f"Error converting data to NumPy arrays: {e}")
        return None, None


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        X: NumPy array of features (images).
        y: NumPy array of labels.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the original dataset to include in the validation split.
        random_state: Seed for the random number generator for reproducibility.

    Returns:
        A tuple containing (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    if X is None or y is None:
        raise ValueError("Input data X or y is None. Cannot split.")
    if not (0 < test_size < 1) or not (0 < val_size < 1) or not (test_size + val_size < 1):
         raise ValueError("test_size and val_size must be between 0 and 1, and their sum must be less than 1.")

    # First split into training+validation and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Ensure proportional class representation
    )

    # Calculate the validation size relative to the remaining data (X_temp)
    # val_size_relative = val_size / (1.0 - test_size)
    # Handle potential floating point inaccuracies or edge cases
    if (1.0 - test_size) <= 0:
         raise ValueError("test_size is too large, no data left for training/validation.")
    val_size_relative = min(max(val_size / (1.0 - test_size), 0.01), 0.99) # Ensure it's a valid proportion


    # Split the temporary set into actual training and validation sets
    # Check if there's enough data left for a meaningful split
    if X_temp.shape[0] < 2 or np.unique(y_temp).size < 2:
         logging.warning("Not enough data or classes in the temporary set for validation split. Returning training and test sets only.")
         # In this case, validation set will be empty or very small depending on split behavior
         # A practical approach might be to just return the temp set as train and skip validation
         X_train, X_val, y_train, y_val = X_temp, np.array([]), y_temp, np.array([])
         # Adjust shapes if empty
         if X_val.size == 0:
             X_val = np.empty((0,) + X_train.shape[1:], dtype=X_train.dtype)
         if y_val.size == 0:
             y_val = np.empty((0,), dtype=y_train.dtype)

    else:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_relative,
                random_state=random_state,
                stratify=y_temp # Stratify again for the validation split
            )
        except ValueError as e:
             logging.warning(f"Could not stratify validation split (likely too few samples per class): {e}. Performing non-stratified split.")
             X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_relative,
                random_state=random_state
            )


    logging.info(f"Data split complete:")
    logging.info(f"  Train set: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"  Validation set: X={X_val.shape}, y={y_val.shape}")
    logging.info(f"  Test set: X={X_test.shape}, y={y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# Example usage:
if __name__ == "__main__":
    logging.info("Starting data loading process...")

    # Create dummy data directories and files for demonstration if they don't exist
    if not os.path.exists(DEFAULT_DATASET_PATH):
        logging.info(f"Creating dummy dataset at: {DEFAULT_DATASET_PATH}")
        os.makedirs(os.path.join(DEFAULT_DATASET_PATH, 'eyes_visible'), exist_ok=True)
        os.makedirs(os.path.join(DEFAULT_DATASET_PATH, 'eyes_away'), exist_ok=True)

        # Create dummy images (simple black/white squares)
        for i in range(50): # Create 50 'visible' images
            img_visible = np.ones((100, 100, 3), dtype=np.uint8) * 255 # White image
            cv2.imwrite(os.path.join(DEFAULT_DATASET_PATH, 'eyes_visible', f'visible_{i}.png'), img_visible)
        for i in range(30): # Create 30 'away' images
            img_away = np.zeros((100, 100, 3), dtype=np.uint8) # Black image
            cv2.imwrite(os.path.join(DEFAULT_DATASET_PATH, 'eyes_away', f'away_{i}.png'), img_away)
        logging.info("Dummy dataset created.")

    # Load the data
    X, y = load_gaze_data(dataset_base_path=DEFAULT_DATASET_PATH)

    if X is not None and y is not None:
        logging.info(f"Total data loaded: X shape={X.shape}, y shape={y.shape}")
        logging.info(f"Label distribution: {np.bincount(y)}")

        # Split the data
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

            logging.info("--- Data Loading and Splitting Summary ---")
            logging.info(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
            logging.info(f"Validation data shape: {X_val.shape}, Validation labels shape: {y_val.shape}")
            logging.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

            # Example: Display one image from each set
            if len(X_train) > 0:
                logging.info(f"Sample Training Image Label: {REVERSE_LABEL_MAP.get(y_train[0], 'Unknown')}")
                # cv2.imshow("Sample Training Image", X_train[0]) # Requires GUI environment
            if len(X_val) > 0:
                 logging.info(f"Sample Validation Image Label: {REVERSE_LABEL_MAP.get(y_val[0], 'Unknown')}")
                # cv2.imshow("Sample Validation Image", X_val[0])
            if len(X_test) > 0:
                 logging.info(f"Sample Test Image Label: {REVERSE_LABEL_MAP.get(y_test[0], 'Unknown')}")
                # cv2.imshow("Sample Test Image", X_test[0])

            # cv2.waitKey(0) # Keep windows open until a key is pressed
            # cv2.destroyAllWindows()

        except ValueError as e:
            logging.error(f"Error during data splitting: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during splitting or display: {e}")
    else:
        logging.error("Data loading failed. Cannot proceed with splitting.")

    logging.info("Data loading script finished.")