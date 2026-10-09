https://leafguard-ai-ranjmffmcobibknvdzz8rq.streamlit.app/
# LeafGuard AI 🌿

A plant leaf disease classifier built with TensorFlow and MobileNetV2. Upload a leaf image to view the predicted class and the model’s top three predictions with confidence scores.

**Live app:** [Try LeafGuard AI](PASTE_YOUR_STREAMLIT_APP_URL_HERE)

## Results

| Metric | Result |
|---|---:|
| Validation accuracy | 93.96% |
| Test accuracy | 93.84% |
| Test loss | 0.1940 |

The test results are from the held-out PlantVillage test set. Real-world photos may perform differently.

## How it works

1. The app accepts a JPG or PNG image.
2. It converts the image to RGB and resizes it to 224 × 224 pixels.
3. The MobileNetV2 model predicts a plant disease class.
4. The app displays the top three predictions and their confidence scores.

The model uses MobileNetV2 transfer learning with ImageNet weights, image augmentation, and a classification layer for the dataset’s classes.

## Project files

```text
leafguard-ai/
├── Models/
│   ├── class_names.json              # Class labels in the model's output order
│   └── leafguard_mobilenetv2.keras   # Trained Keras model
├── Notebook/
│   ├── 01_dataset_analysis.ipynb     # Dataset exploration and visualizations
│   └── 02_model_training.ipynb       # Model training and evaluation
├── reports/
│   └── confusion_matrix.png          # Test-set confusion matrix
├── src/
│   ├── evaluate.py                   # Model evaluation code
│   ├── preprocessing.py              # Image preparation code
│   └── train.py                      # Training script
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files excluded from Git
└── README.md                         # Project documentationg