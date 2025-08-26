# DonorsChoose Project

This project is a web application and API for predicting the success of DonorsChoose.org classroom project proposals. It includes a Flask backend, a user interface, and a Jupyter notebook for data analysis and model development.

## Features
- Flask API for model inference
- Web UI for submitting project details and getting predictions
- Pre-trained machine learning models (XGBoost, SVM, Logistic Regression, Decision Tree)
- Data preprocessing and feature engineering
- Jupyter notebook for exploratory data analysis and model training
- Table of Contents extension support for Jupyter notebooks

## Project Structure
```
DonorChoose_org.ipynb         # Jupyter notebook for analysis and modeling
requirements.txt              # Python dependencies
app.py                        # Flask API backend
ui.py                         # Web UI (Flask)
donorschoose_api/             # API and model files
  models/                     # Pre-trained model and transformer files
  static/                     # Static files (CSS, images)
  templates/                  # HTML templates
model/                        # Additional model files
Raw Data/                     # Raw CSV data files
templates/                    # HTML templates
```

## Setup Instructions
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Enable Jupyter notebook Table of Contents extension:
   ```
   jupyter contrib nbextension install --user
   jupyter nbextension enable toc2/main
   ```
4. Run the Flask API:
   ```
   python app.py
   ```
5. Run the Web UI:
   ```
   python ui.py
   ```
6. Open and explore the Jupyter notebook `DonorChoose_org.ipynb` for data analysis and modeling.

## Notes
- Place your model and transformer `.pkl` files in the appropriate `models/` or `model/` folders.
- Raw data files should be placed in the `Raw Data/` directory.
- The web UI communicates with the API for predictions.

## License
This project is for educational purposes.
