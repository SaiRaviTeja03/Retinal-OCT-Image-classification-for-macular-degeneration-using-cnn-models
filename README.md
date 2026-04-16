# OCT Image Classification for Macular Degeneration Detection

A comprehensive Flask-based web application for classifying retinal OCT (Optical Coherence Tomography) images to detect macular degeneration and related conditions using Convolutional Neural Networks (CNN).

## Features

- **Real-time Image Classification**: Upload OCT images and get instant classification results
- **CNN Model**: Deep learning model trained to classify OCT images into 4 categories:
  - NORMAL
  - CNV (Choroidal Neovascularization)
  - DME (Diabetic Macular Edema)
  - DRUSEN
- **Real-time Alerts**: Live alert system for abnormal findings and system status
- **Interactive Dashboard**: Visual results with confidence scores and statistics
- **Image Preprocessing**: Advanced image enhancement and preprocessing pipeline
- **Drag & Drop Interface**: User-friendly file upload with drag-and-drop support
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
Retinal OCT Image Classification for/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── models/
│   └── cnn_model.py               # CNN model architecture and utilities
├── utils/
│   └── image_processing.py        # Image preprocessing utilities
├── templates/
│   └── index.html                 # Main web interface
├── static/
│   ├── css/
│   │   └── style.css             # Custom styles
│   └── js/
│       └── main.js               # Frontend JavaScript
└── uploads/                       # Uploaded images directory
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project** to your local machine

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv oct_env
   # On Windows
   oct_env\Scripts\activate
   # On Unix/MacOS
   source oct_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Note
Trained model file (.h5) is not included in this repository due to size limitations. Please train the model using the provided training script or use your own dataset.

## Usage

### Uploading and Classifying Images

1. **Upload Method 1 - Drag & Drop**:
   - Drag an OCT image file onto the upload area
   - The system will automatically process and classify the image

2. **Upload Method 2 - File Browser**:
   - Click the "Browse Files" button
   - Select an OCT image from your computer
   - The system will process and classify the image

### Supported File Formats

- PNG
- JPG/JPEG
- TIFF/TIF

### Understanding Results

- **Classification Result**: Shows the predicted class with confidence percentage
- **Confidence Distribution**: Visual chart showing confidence scores for all classes
- **Real-time Alerts**: Get instant notifications for abnormal findings
- **Statistics**: Track total images processed and classification outcomes

### Alert System

The application provides real-time alerts for:
- New image uploads
- Abnormal findings (CNV, DME, DRUSEN)
- System status and errors
- High-confidence abnormal detections

## Model Architecture

The CNN model uses a sequential architecture with:

- **Convolutional Layers**: Multiple Conv2D layers with batch normalization
- **Pooling Layers**: Max pooling for dimensionality reduction
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Softmax activation for multi-class classification

### Model Classes

1. **NORMAL**: Healthy retinal structure
2. **CNV**: Choroidal Neovascularization (wet AMD)
3. **DME**: Diabetic Macular Edema
4. **DRUSEN**: Drusen deposits (dry AMD)

## Image Preprocessing

The application includes advanced image preprocessing:

- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Noise Reduction**: Median filtering and Gaussian blur
- **Normalization**: Pixel value normalization to [0,1] range
- **Resizing**: Standardization to 224x224 pixels
- **Color Space Conversion**: RGB format standardization

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload and classify OCT image
- `GET /alerts`: Get real-time alerts
- `POST /clear_alerts`: Clear all alerts
- `GET /health`: System health check
- `GET /uploads/<filename>`: Access uploaded images

## Configuration

### Model Configuration

You can modify the model architecture in `models/cnn_model.py`:

```python
# Change input size
input_shape = (256, 256, 3)

# Use pretrained models
classifier.create_pretrained_model('ResNet50')
```

### Flask Configuration

Modify Flask settings in `app.py`:

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size
app.config['UPLOAD_FOLDER'] = 'uploads'              # Upload directory
```

## Training Your Own Model

To train the model with your own dataset:

1. **Organize your data**:
   ```
   dataset/
   ├── train/
   │   ├── NORMAL/
   │   ├── CNV/
   │   ├── DME/
   │   └── DRUSEN/
   ├── val/
   │   ├── NORMAL/
   │   ├── CNV/
   │   ├── DME/
   │   └── DRUSEN/
   ```

2. **Create training script**:
   ```python
   from models.cnn_model import OCTClassifier, create_data_generators
   
   # Create classifier
   classifier = OCTClassifier()
   
   # Create data generators
   train_gen, val_gen, _ = create_data_generators(
       'dataset/train', 'dataset/val'
   )
   
   # Train model
   history = classifier.train_model(train_gen, val_gen, epochs=50)
   
   # Save model
   classifier.save_model('models/trained_oct_model.h5')
   ```

## Performance Optimization

### GPU Acceleration

For better performance, use GPU acceleration:

1. **Install TensorFlow with GPU support**:
   ```bash
   pip install tensorflow-gpu
   ```

2. **Verify GPU availability**:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### Production Deployment

For production deployment:

1. **Use Gunicorn**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Use Docker** (optional):
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
   ```

## Troubleshooting

### Common Issues

1. **TensorFlow Installation Error**:
   - Ensure you have the correct Python version (3.8-3.10)
   - Try installing CPU version first: `pip install tensorflow`

2. **Memory Issues**:
   - Reduce batch size in model training
   - Use smaller input image dimensions

3. **File Upload Errors**:
   - Check file size limits
   - Ensure upload directory has write permissions

4. **Model Loading Issues**:
   - Verify model file exists in correct location
   - Check TensorFlow version compatibility

### Debug Mode

Enable debug mode for detailed error messages:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with medical data regulations (HIPAA, GDPR) when using with patient data.

## Disclaimer

This software is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the code documentation
- Create an issue in the project repository

## Acknowledgments

- TensorFlow and Keras for deep learning framework
- Flask for web framework
- Bootstrap for responsive UI components
- Chart.js for data visualization
