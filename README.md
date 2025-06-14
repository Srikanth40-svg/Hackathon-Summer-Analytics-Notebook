# Hackathon-Summer-Analytics-Notebook
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load data
df = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktrain.csv")

# Encode class labels
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Feature engineering function
def create_features(df):
    # Select only NDVI columns (assuming they start with '20')
    ndvi_cols = [col for col in df.columns if col.startswith('20')]
    X = df[ndvi_cols].copy()
    
    # Temporal interpolation for missing values
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=ndvi_cols, index=X.index)
    
    # Smooth the time series
    for col in ndvi_cols:
        X[col] = savgol_filter(X[col], window_length=5, polyorder=2)
    
    # Create temporal features
    features = pd.DataFrame(index=X.index)
    
    # Basic statistics
    features['ndvi_mean'] = X.mean(axis=1)
    features['ndvi_std'] = X.std(axis=1)
    features['ndvi_min'] = X.min(axis=1)
    features['ndvi_max'] = X.max(axis=1)
    features['ndvi_range'] = features['ndvi_max'] - features['ndvi_min']
    
    # Seasonal features (assuming columns are in chronological order)
    n_cols = len(ndvi_cols)
    features['ndvi_slope'] = X.apply(lambda row: np.polyfit(range(n_cols), row, 1)[0], axis=1)
    
    # Phenological metrics (simplified)
    features['ndvi_amplitude'] = X.max(axis=1) - X.min(axis=1)
    
    return features

# Create features
X = create_features(df)
y = df['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Build pipeline with standardization and logistic regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        max_iter=1000,
        class_weight='balanced',
        penalty='l1',
        C=0.1
    )
)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(len(label_encoder.classes_))),
    target_names=label_encoder.classes_
))

# Process test data and create submission
test_data = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktest.csv")
X_test_final = create_features(test_data)
y_test_final = model.predict(X_test_final)
y_decoded = label_encoder.inverse_transform(y_test_final)

result = pd.DataFrame({
    'ID': test_data['ID'],
    'class': y_decoded
})
result.to_csv("improved_submission.csv", index=False)
