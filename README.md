# DERMASCAN: ML-DRIVEN SKIN DISEASE RECOGNITION SYSTEM

DERMADETECT is a machine learning-based system for identifying skin diseases such as chickenpox, measles, and monkeypox using image classification models like KNN, SVM, and AlexNet. The system processes images, extracts features, and classifies them into respective categories to aid in early disease detection.

## Prerequisites

Ensure you have the following installed before proceeding:

- **Python 3.8 or later**
- **pip** (Python package manager)
- **Virtual environment** (optional but recommended)
- **CUDA** (if using GPU acceleration for deep learning models)

### Required Libraries

Before running the project, install the required dependencies using the following command:

```sh
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, manually install the following Python libraries:

```sh
pip install torch torchvision numpy scikit-learn opencv-python flask joblib
```

## Installation Guide

Follow these steps to set up the project on your local system:

1. **Download the Project Files**  
   Ensure you have the necessary project files stored locally on your system.

2. **Clone the Repository**  
   ```sh
   git clone https://github.com/your-repository/DERMADETECT.git
   ```

3. **Navigate to the Project Directory**  
   ```sh
   cd DERMADETECT
   ```

4. **Ensure All Dependencies Are Installed**  
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

To start the application, navigate to the `app` directory and execute the following command:

```sh
cd ./app/
python ./app.py
```

This will start the Flask web server, and you can access the application via `http://127.0.0.1:5000/` in your web browser.

## Project Structure

```
DERMADETECT/
│── app/
│   ├── app.py          # Main Flask application file
│   ├── model.py        # Model loading and prediction script
│   ├── static/         # Contains CSS, JavaScript, and images
│   ├── templates/      # HTML templates for the web application
│   ├── uploads/        # Directory to store uploaded images
│── models/
│   ├── knn_model.pkl   # Pre-trained KNN model
│   ├── svm_model.pkl   # Pre-trained SVM model
│   ├── alexnet_model.pth  # Pre-trained AlexNet model
│── data/
│   ├── dataset/        # Dataset used for training and testing
│── requirements.txt    # Python dependencies
│── README.md           # Documentation
```

## How It Works

1. **Upload an Image**: Users can upload a skin image through the web interface.
2. **Preprocessing**: The system resizes, grayscales, and normalizes the image.
3. **Model Prediction**: The image is analyzed using KNN, SVM, and AlexNet models.
4. **Majority Voting**: The final classification is determined based on the majority vote of the models.
5. **Result Display**: The predicted disease (Chickenpox, Measles, Monkeypox, or Normal) is displayed to the user.

## Model Details

- **K-Nearest Neighbors (KNN)**: Instance-based learning for classification.
- **Support Vector Machine (SVM)**: Linear and non-linear classification algorithm.
- **AlexNet**: Deep convolutional neural network for image classification.

## Troubleshooting

- If you encounter errors related to missing dependencies, rerun:
  ```sh
  pip install -r requirements.txt
  ```
- If the web server does not start, ensure no other application is using port 5000 or specify a different port:
  ```sh
  python app.py --port=8000
  ```

## Future Enhancements

- Improve model accuracy with larger datasets
- Implement additional deep learning models
- Deploy on cloud for better scalability





