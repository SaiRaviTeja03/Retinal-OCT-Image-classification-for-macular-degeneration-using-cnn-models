import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import shutil
from pathlib import Path

from models.cnn_model import OCTClassifier, create_data_generators

def create_synthetic_dataset(base_dir, num_samples_per_class=20):
    """
    Creates a small synthetic dataset of images so the model can be compiled and trained.
    This generates random noise images that resemble OCT scans just enough for the CNN 
    to process them without throwing an error, allowing us to save weights.
    """
    classes = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
    dirs = ['train', 'val']
    
    # Cleanup old data if exists
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    for d in dirs:
        for c in classes:
            dir_path = os.path.join(base_dir, d, c)
            os.makedirs(dir_path, exist_ok=True)
            
            # Generate dummy noise images per class
            num_samples = num_samples_per_class if d == 'train' else num_samples_per_class // 4
            for i in range(num_samples):
                # OCT scans are mostly grayscale noise with some layers
                img_array = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                # Add a simulated retina layer line
                draw = ImageDraw.Draw(img)
                y_pos = 112 + np.random.randint(-20, 20)
                draw.line([(0, y_pos), (224, y_pos)], fill=(200, 200, 200), width=5)
                
                img.save(os.path.join(dir_path, f'synth_oct_{i}.jpg'))
                
    print(f"Generated synthetic dataset at {base_dir}")

def main():
    print("--- Starting OCT Model Training Pipeline ---")
    
    # 1. Create a synthetic dataset in a temporary folder
    data_dir = os.path.join(os.path.dirname(__file__), 'temp_data')
    create_synthetic_dataset(data_dir, num_samples_per_class=30)
    
    # 2. Setup Data Generators
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    print("Loading data generators...")
    train_gen, val_gen, _ = create_data_generators(train_dir, val_dir, batch_size=8, img_size=(224, 224))
    
    # 3. Initialize and compile the Keras CNN
    print("Building and compiling the Convolutional Neural Network...")
    classifier = OCTClassifier(input_shape=(224, 224, 3), num_classes=4)
    classifier.build_model()
    classifier.compile_model()
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # 4. Train the model (Using very few epochs to just compile the weights quickly)
    print("Starting rapid training simulation (3 epochs)...")
    classifier.train_model(
        train_data=train_gen,
        val_data=val_gen,
        epochs=3,
        batch_size=8
    )
    
    # 5. Save the final weights
    model_path = os.path.join('models', 'best_oct_model.h5')
    classifier.save_model(model_path)
    print(f"\nModel successfully trained and weights saved to {model_path}!")
    
    # Cleanup temporary data
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("Cleaned up temporary training data.")
        
    print("--- ✅ Training pipeline completed ---")

if __name__ == "__main__":
    main()
