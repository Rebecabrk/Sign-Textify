import os
import json
import mediapipe as mp
import cv2
import logging
from colorlog import ColoredFormatter

# Configure logging
log_file = "logs/process_coordinates.log"
logger = logging.getLogger("PreprocessingLogger")
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Console handler with color
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
console_handler.setFormatter(formatter)

# Plain text formatter for file
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Input directories
train_dir = "data/asl_alphabet_train"
test_dir = "data/asl_alphabet_test"

# Output directory
output_dir = "processed_coordinates"
train_output_dir = os.path.join(output_dir, "train")
test_output_file = os.path.join(output_dir, "test", "test.json")

# Ensure output directories exist
if not os.path.exists(train_output_dir):
    os.makedirs(train_output_dir)
    logger.info(f"Created directory: {train_output_dir}")
if not os.path.exists(os.path.dirname(test_output_file)):
    os.makedirs(os.path.dirname(test_output_file))
    logger.info(f"Created directory: {os.path.dirname(test_output_file)}")

def process_image(image_path):
    """Process an image to extract hand landmarks and bounding box."""
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"Failed to read image: {image_path}")
        return None

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        logger.info(f"No hands detected in image: {image_path}")
        return None

    # Use the first hand detected
    landmarks = results.multi_hand_landmarks[0]

    # Extract landmarks
    landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]

    # Calculate bounding box
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    bbox = {
        "x_min": min(x_coords),
        "x_max": max(x_coords),
        "y_min": min(y_coords),
        "y_max": max(y_coords),
    }

    logger.debug(f"Processed image: {image_path}")
    return landmarks_list, bbox

def process_directory(data_dir, is_train=True):
    """Process train or test directory."""
    if is_train:
        logger.info(f"Processing train directory: {data_dir}")
        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            if not os.path.isdir(label_path):
                logger.warning(f"Skipping non-directory: {label_path}")
                continue

            label_data = []
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                result = process_image(img_path)
                if result is not None:
                    landmarks, bbox = result
                    label_data.append({
                        "image_name": img_name,
                        "label": label,
                        "bounding_box": bbox,
                        "landmarks": landmarks
                    })

            # Save label data to a JSON file
            output_file = os.path.join(train_output_dir, f"{label}.json")
            with open(output_file, "w") as f:
                json.dump(label_data, f, indent=4)
            logger.info(f"Saved {label} data to {output_file}")
    else:
        logger.info(f"Processing test directory: {data_dir}")
        test_data = []
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            label = os.path.splitext(img_name)[0]  # Label is the image name without extension
            result = process_image(img_path)
            if result is not None:
                landmarks, bbox = result
                test_data.append({
                    "image_name": img_name,
                    "label": label,
                    "bounding_box": bbox,
                    "landmarks": landmarks
                })

        # Save all test data to a single JSON file
        with open(test_output_file, "w") as f:
            json.dump(test_data, f, indent=4)
        logger.info(f"Saved test data to {test_output_file}")

# Process train and test directories
logger.info("Starting preprocessing...")
process_directory(train_dir, is_train=True)
process_directory(test_dir, is_train=False)

# Clean up
hands.close()
logger.info("Processing complete!")
