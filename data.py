# Handles data loading, cleaning, EDA, and preprocessing for Heart Disease dataset
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_heart_data():
    """Fetches and returns the UCI Heart Disease dataset as pandas DataFrame."""
    heart = fetch_ucirepo(id=45)
    # Create DataFrame with features and targets
    df = pd.concat([heart.data.features, heart.data.targets], axis=1)
    logging.info("=" * 50)
    logging.info("UCI HEART DISEASE DATASET LOADED")
    logging.info("=" * 50)
    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"Target column: {df.columns[-1]}")
    logging.info(f"Original target distribution:\n{df[df.columns[-1]].value_counts().sort_index()}")
    return df

def clean_data(df):
    """Handles missing values and converts target to binary (0 = no disease, 1 = disease present)."""
    df_clean = df.copy()
    missing_before = df_clean.isnull().sum().sum()
    logging.info(f"Missing values before cleaning: {missing_before}")
    
    # Fill missing values with column median
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    
    # Convert target to binary: 0 = no disease, 1-4 = disease present → 1
    target_col = df_clean.columns[-1]
    df_clean[target_col] = (df_clean[target_col] > 0).astype(int)
    
    logging.info(f"Converted to binary. New distribution:\n{df_clean[target_col].value_counts()}")
    logging.info(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
    return df_clean

def get_feature_descriptions():
    """Returns user-friendly descriptions for each feature."""
    return {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'cp': 'Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)',
        'trestbps': 'Resting blood pressure (in mm Hg on admission to the hospital)',
        'chol': 'Serum cholesterol in mg/dl',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'restecg': 'Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1 = yes; 0 = no)',
        'oldpeak': 'ST depression induced by exercise relative to rest',
        'slope': 'Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)',
        'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
        'thal': 'Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)',
    }

def get_simple_questions():
    """Returns simplified questions for patients with CORRECT feature names matching the dataset."""
    return [
        {
            'feature': 'age',
            'question': 'What is your age?',
            'type': 'number',
            'min': 20,
            'max': 100,
            'default': 54,
            'unit': 'years',
            'icon': '👤',
            'category': 'Demographics',
        },
        {
            'feature': 'sex',
            'question': 'What is your gender?',
            'type': 'select',
            'options': {'Female': 0, 'Male': 1},
            'icon': '⚥',
            'category': 'Demographics',
        },
        {
            'feature': 'cp',
            'question': 'What type of chest pain do you experience?',
            'type': 'select',
            'options': {
                'Typical angina': 1,
                'Atypical angina': 2,
                'Non-anginal pain': 3,
                'Asymptomatic': 4,
            },
            'icon': '❤️‍🩹',
            'category': 'Symptoms',
        },
        {
            'feature': 'trestbps',
            'question': 'What is your resting blood pressure?',
            'type': 'number',
            'min': 80,
            'max': 200,
            'default': 130,
            'unit': 'mm Hg',
            'icon': '💓',
            'category': 'Vitals',
        },
        {
            'feature': 'chol',
            'question': 'What is your cholesterol level?',
            'type': 'number',
            'min': 100,
            'max': 600,
            'default': 240,
            'unit': 'mg/dl',
            'icon': '🧪',
            'category': 'Lab Results',
        },
        {
            'feature': 'fbs',
            'question': 'Is your fasting blood sugar greater than 120 mg/dl?',
            'type': 'select',
            'options': {'No': 0, 'Yes': 1},
            'icon': '🩸',
            'category': 'Lab Results',
        },
        {
            'feature': 'restecg',
            'question': 'What were your resting ECG results?',
            'type': 'select',
            'options': {
                'Normal': 0,
                'ST-T Wave Abnormality': 1,
                'Left Ventricular Hypertrophy': 2,
            },
            'icon': '📊',
            'category': 'ECG',
        },
        {
            'feature': 'thalach',
            'question': 'What is your maximum heart rate achieved?',
            'type': 'number',
            'min': 60,
            'max': 220,
            'default': 150,
            'unit': 'bpm',
            'icon': '💪',
            'category': 'Exercise',
        },
        {
            'feature': 'exang',
            'question': 'Do you experience exercise-induced angina?',
            'type': 'select',
            'options': {'No': 0, 'Yes': 1},
            'icon': '🏃',
            'category': 'Symptoms',
        },
        {
            'feature': 'oldpeak',
            'question': 'What is your ST depression induced by exercise?',
            'type': 'number',
            'min': 0,
            'max': 6,
            'default': 1.5,
            'step': 0.1,
            'unit': 'mm',
            'icon': '📉',
            'category': 'Exercise',
        },
        {
            'feature': 'slope',
            'question': 'What is the slope of your peak exercise ST segment?',
            'type': 'select',
            'options': {
                'Upsloping': 1,
                'Flat': 2,
                'Downsloping': 3,
            },
            'icon': '📈',
            'category': 'Exercise',
        },
        {
            'feature': 'ca',
            'question': 'How many major vessels are colored by fluoroscopy?',
            'type': 'select',
            'options': {
                '0 vessels': 0,
                '1 vessel': 1,
                '2 vessels': 2,
                '3 vessels': 3,
            },
            'icon': '🫀',
            'category': 'Imaging',
        },
        {
            'feature': 'thal',
            'question': 'What is your thalassemia result?',
            'type': 'select',
            'options': {
                'Normal': 3,
                'Fixed Defect': 6,
                'Reversible Defect': 7,
            },
            'icon': '🧬',
            'category': 'Imaging',
        },
    ]

def preprocess_data(df, test_size=0.2, random_state=42, n_components=8):
    """
    Splits, scales, and applies PCA.
    IMPORTANT: n_components defaults to 8 (the dataset has 13 features so 8 is safely below that limit).
    app.py's get_trained_pipeline() always calls this function once and reuses the returned (scaler, pca)
    objects for inference, so the number of PCA components is always consistent.
    Returns: X_train_pca, X_test_pca, y_train, y_test, scaler, pca
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    n_features = X.shape[1]
    # Guard: n_components must be <= number of features
    n_components = min(n_components, n_features)
    
    logging.info(f"Features ({n_features}): {X.columns.tolist()}")
    logging.info(f"Target distribution: {y.value_counts().to_dict()}")
    logging.info(f"PCA n_components: {n_components}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    logging.info(
        f"Explained variance with {n_components} components: "
        f"{pca.explained_variance_ratio_.sum():.3f}"
    )
    
    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca

def verify_dataset(df):
    """Verify that the dataset has the expected structure."""
    expected_features = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
    ]
    actual_features = df.columns[:-1].tolist()
    
    logging.info("\n" + "=" * 50)
    logging.info("DATASET VERIFICATION")
    logging.info("=" * 50)
    logging.info(f"Expected: {expected_features}")
    logging.info(f"Actual: {actual_features}")
    
    missing = set(expected_features) - set(actual_features)
    extra = set(actual_features) - set(expected_features)
    
    if missing:
        logging.warning(f" Missing features: {missing}")
    else:
        logging.info(" All expected features present")
    
    if extra:
        logging.warning(f"Extra features found: {extra}")
    
    for col in actual_features:
        logging.info(
            f" {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}"
        )
    
    return len(missing) == 0