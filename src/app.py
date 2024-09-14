import os
import pandas as pd
import pytesseract
import cv2
from PIL import Image
import re
from utils import parse_string, common_mistake, download_image
from sanity import sanity_check
import constants
import multiprocessing
from tqdm import tqdm
from functools import partial

# Paths to dataset files
TRAIN_CSV = 'dataset/train.csv'
TEST_CSV = 'dataset/test.csv'
OUTPUT_CSV = 'dataset/test_out.csv'
DOWNLOAD_FOLDER = 'dataset/images'

def preprocess_image(image_path):
    """Preprocess the image to make it more suitable for OCR."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, img_thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    return img_thresh

def extract_entity_value(image_path):
    """Use OCR to extract entity values from the image."""
    img = preprocess_image(image_path)
    if img is None:
        return ""
    ocr_result = pytesseract.image_to_string(img)
    
    # Pattern to extract a number followed by a unit (like "25 gram", "12.5 cm")
    pattern = r"(\d+(\.\d+)?)\s?([a-zA-Z]+)"
    match = re.search(pattern, ocr_result)
    if match:
        number = match.group(1)
        unit = match.group(3)
        try:
            number = float(number)
            unit = unit.lower().strip()
            unit = common_mistake(unit)
            if unit in constants.allowed_units:
                return f"{number} {unit}"
        except Exception as e:
            print(f"Error parsing OCR result: {ocr_result}. Error: {e}")
            return ""
    return ""

def download_images_with_limit(image_links, download_folder, max_workers=60):
    """Download images while limiting the number of multiprocessing workers."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_image_partial = partial(download_image, save_folder=download_folder, retries=3, delay=3)
    
    with multiprocessing.Pool(max_workers) as pool:
        list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
        pool.close()
        pool.join()

def generate_predictions(test_df):
    """Generate predictions for the test dataset."""
    predictions = []
    print("Downloading images...")
    download_images_with_limit(test_df['image_link'], DOWNLOAD_FOLDER, max_workers=60)

    print("Extracting entity values from images...")
    for idx, row in test_df.iterrows():
        image_name = os.path.basename(row['image_link'])
        image_path = os.path.join(DOWNLOAD_FOLDER, image_name)
        prediction = extract_entity_value(image_path)
        try:
            parse_string(prediction)
            predictions.append(prediction)
        except ValueError as ve:
            print(f"Prediction format error at index {row['index']}: {ve}")
            predictions.append("")  # If prediction format is invalid, append an empty string.
    return predictions

def main():
    """Main function to run the prediction process."""
    test_df = pd.read_csv(TEST_CSV)
    if 'index' not in test_df.columns or 'image_link' not in test_df.columns:
        raise ValueError("Test CSV file must contain 'index' and 'image_link' columns.")
    
    test_df['prediction'] = generate_predictions(test_df)
    
    # Save predictions to output CSV
    output_df = test_df[['index', 'prediction']]
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    print("Running sanity check...")
    try:
        sanity_check(TEST_CSV, OUTPUT_CSV)
        print(f"Predictions successfully saved in {OUTPUT_CSV} and passed the sanity check!")
    except Exception as e:
        print(f"Sanity check failed: {e}")

if __name__ == "__main__":
    main()
