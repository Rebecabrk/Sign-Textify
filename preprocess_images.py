import os
import cv2
import logging
import mediapipe as mp
from stats import StatsGenerator
import colorlog

# Configure colorlog for console output
console_handler = colorlog.StreamHandler()
console_handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

# Configure logging for file output
file_handler = logging.FileHandler('preprocess.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))

# Get the logger and set the level
logger = colorlog.getLogger()
logger.setLevel(logging.INFO)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("Starting preprocess...")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.4)
mp_drawing = mp.solutions.drawing_utils

# Paths
dataset_paths = ["data/asl_alphabet_train", "data/asl_alphabet_test"]
output_paths = ["processed_dataset/train", "processed_dataset/test"]

# Create output directories
for output_path in output_paths:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info("Created directory: %s", output_path)
    else:
        logger.info("Directory already exists: %s", output_path)

# Preprocessing function
def preprocess_image(image_path, output_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        logger.info("Skipping image:", image_path, "could not load image")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 50    # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        results = hands.process(adjusted)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks (optional visualization step)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Crop and resize hand region
            h, w, _ = image.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            cropped = image[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                logger.warning("Skipping image:", image_path, "crop size is 0")
                return None
            resized = cv2.resize(cropped, output_size)
            
            return resized
    else:
        return None  # Skip images without detected hands

# Process each dataset path
for dataset_path, output_path in zip(dataset_paths, output_paths):
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        output_label_path = os.path.join(output_path, label)
        
        if not os.path.exists(output_label_path):
            os.makedirs(output_label_path)
            logger.info("Created directory for label: %s", output_label_path)
        
        elif len(os.listdir(output_label_path)) == len(os.listdir(label_path)):
            logger.warning("Skipping label: %s as it already exists in the output directory", label)
            continue
        
        logger.info("Preprocessing for label: %s", label)
        
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            logger.info("Processing image: %s", image_path)
            processed_image = preprocess_image(image_path)
            if processed_image is not None:
                output_image_path = os.path.join(output_label_path, image_file)
                cv2.imwrite(output_image_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                logger.info("Saved processed image to: %s", output_image_path)
            else:
                logger.warning("Skipping image: %s as no hands were detected", image_file)

logger.info("Preprocess completed.")

# Genarate stats
original_train_path = 'data/asl_alphabet_train'
original_test_path = 'data/asl_alphabet_test'
preprocessed_train_data = 'processed_dataset/train'
preprocessed_test_data = 'processed_dataset/test'
output_file = 'stats.md'
stats_generator = StatsGenerator(original_train_path, preprocessed_train_data, original_test_path, preprocessed_test_data, output_file)
stats_generator.generate_stats()
