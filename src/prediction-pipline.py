import os
import warnings
import sys
import math
import matplotlib.pyplot as plt

# 1. Silence background noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import tensorflow_hub as hub
import tf_keras as keras
import numpy as np
import json
import argparse
from glob import glob

IMG_SIZE = 224

def process_image(image_path):
    """Load and preprocess image. Returns tensor and original numpy array."""
    try:
        raw_img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(raw_img, channels=3)
        original = img.numpy()
        
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=[IMG_SIZE, IMG_SIZE])
        return tf.expand_dims(img, axis=0), original
    except Exception as e:
        print(f"\n[!] Error processing {os.path.basename(image_path)}: {e}")
        return None, None

def display_results_grid(results):
    """Generate a visual summary gallery of all predictions."""
    if not results:
        return
    
    num_images = len(results)
    cols = min(num_images, 3) # Max 3 columns for better visibility
    rows = math.ceil(num_images / cols)
    
    plt.figure(figsize=(12, 4 * rows))
    plt.suptitle("Dog Breed Classification Results", fontsize=16, fontweight='bold', y=0.98)
    
    for i, (img, breed, conf, fname) in enumerate(results):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        color = 'green' if conf > 70 else 'orange'
        plt.title(f"File: {fname}\n{breed.upper()}\n{conf:.2f}%", 
                  fontsize=10, color=color, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("\n[INFO] Gallery window opened. Close it to finish execution.")
    plt.show()

class DogBreedPredictor:
    """Handles model loading and inference logic."""
    def __init__(self, model_path, labels_path):
        print(f"\n[1/2] Initializing model... ", end="", flush=True)
        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            print("Failed!")
            print(f"[ERROR] Missing files: {model_path} or {labels_path}")
            sys.exit(1)

        try:
            self.model = keras.models.load_model(
                model_path, 
                custom_objects={'KerasLayer': hub.KerasLayer}
            )
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
            print("Done!")
        except Exception as e:
            print(f"Failed! \n[ERROR] {e}")
            sys.exit(1)

    def predict(self, img_tensor):
        """Run image through the model and return prediction data."""
        probs = self.model.predict(img_tensor, verbose=0)
        idx = np.argmax(probs)
        return self.labels[idx], np.max(probs) * 100

def main():
    """Main pipeline execution flow."""
    parser = argparse.ArgumentParser(description="Dog Breed Classification Pipeline")
    parser.add_argument("--input_dir", default="images_to_predict/")
    parser.add_argument("--model", default="models/mobilenet_v2_dog_breed_classifier.h5")
    parser.add_argument("--labels", default="models/labels.json")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        os.makedirs(args.input_dir)
        print(f"Directory '{args.input_dir}' created. Add images and restart.")
        return

    predictor = DogBreedPredictor(args.model, args.labels)
    image_files = glob(os.path.join(args.input_dir, "*.jp*g"))
    
    if not image_files:
        print(f"\n[!] No images found in '{args.input_dir}'.")
        return

    print(f"[2/2] Running inference on {len(image_files)} images...\n")
    print(f"{'FILENAME':<25} | {'PREDICTED BREED':<25} | {'CONFIDENCE':<10}")
    print("-" * 65)

    all_results = []

    for f in image_files:
        img_tensor, original_img = process_image(f)
        if img_tensor is not None:
            breed, conf = predictor.predict(img_tensor)
            fname = os.path.basename(f)
            print(f"{fname[:23]:<25} | {breed.upper():<25} | {conf:>8.2f}%")
            all_results.append((original_img, breed, conf, fname))
    
    print("\n--- Pipeline Completed ---")
    
    # Auto-display the gallery at the end of every run
    if all_results:
        display_results_grid(all_results)

if __name__ == "__main__":
    main()