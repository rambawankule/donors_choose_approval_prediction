import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
import re
import string
from collections import Counter
import scipy.sparse # Used for combining sparse TF-IDF matrices and dense numerical/binary features [6]

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Global Variables for Model and Preprocessing Objects ---
# These will be loaded once when the application starts
xgb_model = None
scaler = None
categorical_transformer = None
text_transformer_essay = None
text_transformer_title = None
text_transformer_resource_summary = None

# Define winsorization thresholds (these would ideally be saved during training)
# In a real-world scenario, these values (e.g., 95th percentile from training data)
# would be determined and persisted during the model training phase.
# For this example, placeholder values are used based on general observations.
WINZORIZED_TOTAL_COST_THRESHOLD = 1500.0 # Example based on a typical 95th percentile [7]
WINZORIZED_QUANTITY_THRESHOLD = 10.0     # Example based on a typical 95th percentile [8]

# Define STOPWORDS for text preprocessing [9]
STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", \
             "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
             'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
             'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
             'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
             'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
             "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
             "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
             'won', "won't", 'wouldn', "wouldn't"]


# --- 3. Load Model and Preprocessing Objects (executed once on startup) ---
# This function loads all necessary serialized assets when the Flask app starts. [10, 11]
def load_assets():
    global xgb_model, scaler, categorical_transformer, \
           text_transformer_essay, text_transformer_title, text_transformer_resource_summary

    # Adjust path based on the directory structure provided
    models_path = os.path.join(os.path.dirname(__file__), 'models')

    try:
        xgb_model = joblib.load(os.path.join(models_path, 'xgb_model.pkl'))
        scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
        categorical_transformer = joblib.load(os.path.join(models_path, 'categorical_encoder.pkl'))
        text_transformer_essay = joblib.load(os.path.join(models_path, 'essay_tfidf.pkl'))
        text_transformer_title = joblib.load(os.path.join(models_path, 'title_tfidf.pkl'))
        text_transformer_resource_summary = joblib.load(os.path.join(models_path, 'resource_summary_tfidf.pkl'))
        app.logger.info("All models and preprocessing objects loaded successfully.")
    except Exception as e:
        app.logger.error(f"Error loading assets: {e}")
        # If critical assets fail to load, the application cannot function.
        # It's better to raise an error to prevent the server from starting incorrectly.
        raise

# --- 4. Preprocessing Functions (Mirroring Data Processing and Feature Engineering from sources) ---

# Function to decontract common English contractions [12]
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Function for general text preprocessing (decontract, remove special chars, stopwords, lowercase) [13]
def preprocess_text(text):
    text = str(text) # Ensure it's a string
    text = decontracted(text)
    text = text.replace('\\r', ' ').replace('\\"', ' ').replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in STOPWORDS)
    return text.lower().strip()

# Function to clean subject categories and subcategories [14, 15]
def clean_categories_subcategories(category_string):
    if pd.isna(category_string):
        return ""
    temp = ""
    for j in category_string.split(','):
        if 'The' in j.split():
            j = j.replace('The', '')
        j = j.replace(' ', '')
        temp += j.strip() + " "
    temp = temp.replace('&', '_')
    return temp.strip()

# Function to categorize resources based on keywords in their description [16, 17]
def categorize_resource(description):
    description = str(description).lower()
    if any(keyword in description for keyword in ['computer', 'laptop', 'tablet', 'chromebook', 'ipad', 'software', 'technology', 'digital', 'printer', 'scanner']):
        return 'Technology'
    elif any(keyword in description for keyword in ['book', 'library', 'reading', 'novel', 'atlas', 'dictionary']):
        return 'Books'
    elif any(keyword in description for keyword in ['supply', 'paper', 'pencil', 'pen', 'marker', 'crayon', 'glue', 'scissor', 'notebook', 'folder', 'art', 'craft', 'material']):
        return 'Supplies'
    elif any(keyword in description for keyword in ['sport', 'ball', 'equipment', 'jersey', 'physical education', 'health']):
        return 'Sports & Health'
    elif any(keyword in description for keyword in ['science', 'lab', 'experiment', 'microscope', 'beaker', 'chemistry', 'biology', 'physics']):
        return 'Science'
    elif any(keyword in description for keyword in ['math', 'mathematics', 'calculator', 'geometry', 'algebra']):
        return 'Math'
    elif any(keyword in description for keyword in ['music', 'instrument', 'band', 'orchestra', 'choir', 'art', 'paint', 'canvas', 'drawing']):
        return 'Music & Arts'
    elif any(keyword in description for keyword in ['furniture', 'desk', 'chair', 'table', 'storage', 'shelf']):
        return 'Furniture'
    else:
        return 'Other'

# Main preprocessing function that mirrors the training pipeline
def preprocess_input(data):
    # Convert input JSON to a pandas DataFrame row for consistent processing
    df = pd.DataFrame([data])

    # --- Handle Missing Values [18, 19] ---
    df['project_essay_3'] = df['project_essay_3'].fillna('')
    df['project_essay_4'] = df['project_essay_4'].fillna('')
    # teacher_prefix was filled with mode in training. For single inference, use a default mode ('Ms.') if missing.
    if 'teacher_prefix' not in df.columns or df['teacher_prefix'].isnull().all():
        df['teacher_prefix'] = 'Ms.'

    # --- Data Type Conversions and Date Feature Extraction [20-22] ---
    df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime'], errors='coerce')
    df['submission_hour'] = df['project_submitted_datetime'].dt.hour
    df['submission_day_of_week'] = df['project_submitted_datetime'].dt.day_name()
    df['submission_month'] = df['project_submitted_datetime'].dt.month

    # --- Essay Preprocessing and Feature Engineering [13, 23-25] ---
    df['combined_essays'] = df['project_essay_1'].fillna('') + ' ' + \
                            df['project_essay_2'].fillna('') + ' ' + \
                            df['project_essay_3'].fillna('') + ' ' + \
                            df['project_essay_4'].fillna('')
    df['essay'] = df['combined_essays'].apply(preprocess_text)
    df['essay_length'] = df['combined_essays'].apply(lambda x: len(str(x).split()))

    # Keyword-based binary features from essay content [25]
    keywords = ['innovation', 'community', 'support', 'students', 'learning']
    for keyword in keywords:
        df[f'has_{keyword}'] = df['essay'].apply(lambda x: 1 if keyword in x else 0)

    # --- Project Title Preprocessing [26] ---
    df['project_title'] = df['project_title'].apply(preprocess_text)

    # --- Project Resource Summary Preprocessing [27] ---
    df['project_resource_summary_preprocessed'] = df['project_resource_summary'].apply(preprocess_text)

    # --- Project Grade Category Preprocessing [28] ---
    df['project_grade_category'] = df['project_grade_category'].apply(lambda x: str(x).replace(' ', '_').replace('-','_to_').lower())

    # --- Subject Categories Cleaning [14, 15] ---
    df['clean_categories'] = df['project_subject_categories'].apply(clean_categories_subcategories)
    df['clean_subcategories'] = df['project_subject_subcategories'].apply(clean_categories_subcategories)

    # --- Resource Aggregation and Feature Engineering (from `resources` list in input JSON) ---
    # Extract resources list for this single-row DataFrame. Expect a list of dicts at row 0.
    resources_list = []
    if 'resources' in df.columns and len(df['resources']) > 0:
        resources_list = df['resources'].iloc[0]
    if resources_list is None:
        resources_list = []
    # If a single resource dict was provided, wrap it in a list
    if isinstance(resources_list, dict):
        resources_list = [resources_list]

    # Create DataFrame from list of resource dicts. If empty, create a minimal default row
    if len(resources_list) == 0:
        resources_df_single = pd.DataFrame([{'quantity': 0, 'price': 0.0, 'description': 'No Description'}])
    else:
        resources_df_single = pd.DataFrame(resources_list)

    # Ensure description column exists and fill missing descriptions
    if 'description' not in resources_df_single.columns:
        resources_df_single['description'] = 'No Description'
    else:
        resources_df_single['description'].fillna('No Description', inplace=True)

    # Calculate total cost and total quantity from resources list [29]
    resources_df_single['total_cost_per_item'] = resources_df_single['quantity'] * resources_df_single['price']
    total_cost_project = resources_df_single['total_cost_per_item'].sum()
    total_quantity_project = resources_df_single['quantity'].sum()

    # Apply winsorization using predefined thresholds [7, 8]
    df['total_cost_winsorized'] = min(total_cost_project, WINZORIZED_TOTAL_COST_THRESHOLD)
    df['quantity_winsorized'] = min(total_quantity_project, WINZORIZED_QUANTITY_THRESHOLD)

    # Resource category binary features [3, 16, 30, 31]
    resources_df_single['resource_category'] = resources_df_single['description'].apply(categorize_resource)
    project_resource_categories_list = resources_df_single['resource_category'].unique().tolist()

    # Define the *exact* list of binary features used in training for `X_binary` in `X_combined` [3]
    # This list ensures consistency with the model's expected input features.
    binary_features_for_model = [
        'has_innovation', 'has_community', 'has_support', 'has_students', 'has_learning',
        'has_supplies', 'has_books', 'has_furniture', 'has_other', 'has_math',
        'has_technology'
    ]
    # Create these binary features, some are keyword-based from essay, some are resource-category based
    for category_feat_name in binary_features_for_model[5:]: # Start from 'has_supplies'
        # Convert feature name back to category name (e.g., 'has_supplies' -> 'Supplies')
        category_name_from_feat = category_feat_name.replace('has_', '').replace('_', ' ').title()
        # Special handling for 'Other' as it is title cased from categorize_resource
        if category_name_from_feat == 'Other':
             df[category_feat_name] = 1 if 'Other' in project_resource_categories_list else 0
        else:
             df[category_feat_name] = 1 if category_name_from_feat in project_resource_categories_list else 0


    # --- Select and prepare features for model input (maintaining order) [6, 32, 33] ---
    numerical_features_to_scale = [
        'teacher_number_of_previously_posted_projects',
        'essay_length',
        'total_cost_winsorized',
        'quantity_winsorized',
        'submission_hour',
        'submission_month'
    ]
    categorical_features_to_encode = [
        'teacher_prefix',
        'school_state',
        'project_grade_category',
        'clean_categories',
        'clean_subcategories',
        'submission_day_of_week'
    ]
    text_features_to_vectorize = ['essay', 'project_title', 'project_resource_summary_preprocessed']


    # Extract raw feature data for transformation
    X_numerical_raw_for_scaling = df[numerical_features_to_scale].values
    X_categorical_raw_for_encoding = df[categorical_features_to_encode]
    X_binary_raw_for_model = df[binary_features_for_model].values

    # For single-row inference take the scalar string values
    X_text_essay_raw = df['essay'].iloc[0]
    X_text_title_raw = df['project_title'].iloc[0]
    X_text_resource_summary_raw = df['project_resource_summary_preprocessed'].iloc[0]


    # --- Apply loaded transformers to the raw data ---
    X_numerical_scaled = scaler.transform(X_numerical_raw_for_scaling)
    X_categorical_encoded = categorical_transformer.transform(X_categorical_raw_for_encoding)
    # Transformers expect an iterable of documents (list-like)
    essay_tfidf_transformed = text_transformer_essay.transform([X_text_essay_raw])
    title_tfidf_transformed = text_transformer_title.transform([X_text_title_raw])
    resource_summary_tfidf_transformed = text_transformer_resource_summary.transform([X_text_resource_summary_raw])

    # Combine all processed features into a single sparse matrix, matching training order [6]
    # This horizontal stacking must exactly replicate the order of features used during model training.
    X_final = scipy.sparse.hstack([
        X_numerical_scaled,
        X_categorical_encoded,
        essay_tfidf_transformed,
        title_tfidf_transformed,
        resource_summary_tfidf_transformed,
        X_binary_raw_for_model
    ]).tocsr() # Convert to CSR format for efficient slicing and operations

    return X_final


# --- 5. Flask API Endpoints ---
@app.route('/')
def home():
    """Basic endpoint to confirm the API is running."""
    return "DonorsChoose.org Project Approval Prediction API is running!"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify if the model and preprocessors are loaded.""" [34]
    if all([xgb_model, scaler, categorical_transformer,
            text_transformer_essay, text_transformer_title, text_transformer_resource_summary]):
        return jsonify({'status': 'Model and preprocessors loaded and ready!'}), 200
    else:
        return jsonify({'status': 'Error: Model or preprocessors not loaded!'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint that accepts project data in JSON format,
    preprocesses it, and returns the predicted approval status. [10]
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)

    # Basic input validation for critical fields required for preprocessing
    required_fields = [
        "teacher_prefix", "school_state", "project_submitted_datetime",
        "project_grade_category", "project_subject_categories",
        "project_subject_subcategories", "project_title",
        "project_essay_1", "project_essay_2",
        "project_resource_summary",
        "teacher_number_of_previously_posted_projects",
        "resources" # This is expected to be a list of resource dictionaries
    ]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Ensure optional essay fields are present, even if empty, as expected by preprocessing
    data.setdefault("project_essay_3", "")
    data.setdefault("project_essay_4", "")

    try:
        # Preprocess the incoming request data
        processed_input = preprocess_input(data)

        # Make prediction using the loaded XGBoost model
        prediction_proba = xgb_model.predict_proba(processed_input)[:, 1]
        
        # Apply a threshold (e.g., 0.5) to convert probability to binary prediction
        prediction = (prediction_proba >= 0.5).astype(int)
        
        predicted_status = "Approved" if prediction == 1 else "Rejected"

        return jsonify({
            "predicted_approval": predicted_status,
            "confidence_score": float(prediction_proba) # Return confidence as well
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500

# --- 6. Run the Flask App ---
if __name__ == "__main__":
    # Load model and preprocessors when the app starts, not for every request. [10, 35]
    load_assets() 
    # Run the Flask app in debug mode; host '0.0.0.0' makes it accessible externally.
    app.run(debug=True, host='0.0.0.0')