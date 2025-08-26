# DonorsChoose Project
<img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/b417c5ca-f789-440c-a6a3-eef767583010" />

This project is a web application and API for predicting the success of DonorsChoose.org classroom project proposals. It includes a Flask backend, a user interface, and a Jupyter notebook for data analysis and model development.

## Features
- Flask API for model inference
- Web UI for submitting project details and getting predictions
- Pre-trained machine learning models (XGBoost, SVM, Logistic Regression, Decision Tree)
- Data preprocessing and feature engineering
- Jupyter notebook for exploratory data analysis and model training

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
templates/                    # HTML templates
```

## Setup Instructions
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Flask API:
   ```
   python app.py
   ```
4. Run the Web UI:
   ```
   python ui.py
   ```
5. Open and explore the Jupyter notebook `DonorChoose_org.ipynb` for data analysis and modeling.

## License
This project is for educational purposes.
