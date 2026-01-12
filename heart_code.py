import os  # Add this line
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, classification_report,
                            confusion_matrix, ConfusionMatrixDisplay)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path):
    import os

def load_data(path):
    """Load and return the dataset."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        logger.info("Data loaded successfully. Shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise
    """Load and return the dataset."""
    try:
        df = pd.read_csv(path)
        logger.info("Data loaded successfully. Shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

def preprocess_data(df, target_col='target'):
    """
    Preprocess data: Split features/target, scale, and split train/test.
    """
    try:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error("Error in preprocessing: %s", e)
        raise

def train_model(model, X_train, y_train, param_grid=None):
    """Train a model with optional hyperparameter tuning."""
    try:
        if param_grid:
            logger.info("Performing hyperparameter tuning for %s", model.__class__.__name__)
            grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)
            logger.info("Best params: %s", grid.best_params_)
            return grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model
    except Exception as e:
        logger.error("Error training %s: %s", model.__class__.__name__, e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate and return metrics for a trained model."""
    try:
        y_pred = model.predict(X_test)
        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
    except Exception as e:
        logger.error("Error evaluating model: %s", e)
        raise

def plot_metrics(results_df):
    """Generate bar chart and pie chart of model metrics."""
    try:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        models = results_df['Model'].tolist()
        numeric_data = results_df[metrics].values
        
        # Bar Chart
        plt.figure(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, numeric_data[:, i], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width*1.5, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Pie Chart (Accuracy)
        plt.figure(figsize=(8, 8))
        plt.pie(
            results_df['Accuracy'],
            labels=models,
            autopct='%1.1f%%',
            startangle=140,
            shadow=True,
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']
        )
        plt.title('Accuracy Distribution Across Models')
        plt.show()
    except Exception as e:
        logger.error("Error generating plots: %s", e)

def save_model(model, filename):
    """Save trained model to disk."""
    try:
        dump(model, filename)
        logger.info("Model saved as %s", filename)
    except Exception as e:
        logger.error("Error saving model: %s", e)

def main():
    """Main function to train and evaluate models."""
    # Load and preprocess data
    df = load_data("heart.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Define models and their parameter grids for tuning
    models = {
        'SVM': {
            'model': SVC(),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        },
        'Logistic Regression': {
            'model': LogisticRegression(),
            'params': {'C': [0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {'n_estimators': [50, 100, 200]}
        },
        'XGBoost': {
            'model': XGBClassifier(),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7]}
        }
    }
    
    # Train, evaluate, and store results
    results = []
    for name, config in models.items():
        logger.info("\nTraining %s...", name)
        
        # Train with hyperparameter tuning
        model = train_model(config['model'], X_train, y_train, config['params'])
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        results.append([name] + list(metrics.values()))
        
        # Save model
        save_model(model, f"{name.lower().replace(' ', '_')}_model.joblib")
        
        # Print classification report
        y_pred = model.predict(X_test)
        logger.info("\n%s Performance:\n%s", name, classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(f'Confusion Matrix - {name}')
        plt.show()
    
    # Create results dataframe
    results_df = pd.DataFrame(
        results,
        columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    )
    
    # Display results and plots
    print("\n=== Final Model Comparison ===")
    print(results_df.to_markdown(index=False))
    
    # Generate visualizations
    plot_metrics(results_df)

if __name__ == "__main__":
    main()
    
def predict_new_patient():
    """Interactive function to predict heart disease risk"""
    try:
        model = joblib.load('model/heart_disease_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return
    
    print("\nEnter patient details:")
    inputs = [
        int(input("Age: ")),
        int(input("Sex (1=Male, 0=Female): ")),
        int(input("Chest Pain Type (0-3): ")),
        int(input("Resting BP (mmHg): ")),
        int(input("Cholesterol (mg/dl): ")),
        int(input("Fasting BS >120? (1=Yes, 0=No): ")),
        int(input("Resting ECG (0-2): ")),
        int(input("Max Heart Rate: ")),
        int(input("Exercise Angina (1=Yes, 0=No): ")),
        float(input("ST Depression (0.0-6.2): ")),
        int(input("Slope (0-2): ")),
        int(input("Major Vessels (0-3): ")),
        int(input("Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect): "))
    ]
    
    # Create DataFrame with correct feature names
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                    'restecg', 'thalach', 'exang', 'oldpeak', 
                    'slope', 'ca', 'thal']
    new_patient = pd.DataFrame([inputs], columns=feature_names)
    
    # Scale features
    new_patient_scaled = scaler.transform(new_patient)
    
    # Make prediction
    prediction = model.predict(new_patient_scaled)
    probability = model.predict_proba(new_patient_scaled)[:, 1][0]
    # Display results
    result = "Heart Disease" if prediction[0] else "No Heart Disease"
    print(f"\n🩺 Prediction: {result} (Confidence: {probability:.1%})")

#     if __name__ == "__main__":
# #     # Train models and generate reports
#      results_df = train_and_evaluate_models() 
  
    
#      print("\n✅ Training complete! Results saved to:")
#      print("- README.md")
#      print("- results/ directory")
#     print("- model/ directory (saved model and scaler)")
    
#     print("\nModel Performance Summary:")
#     print(results_df.sort_values('ROC-AUC', ascending=False).to_markdown(index=False))
    
# #     # Run interactive prediction
#     print("\nStarting prediction interface...")
    
    
#     predict_new_patient()
    
    if __name__ == "__main__":
     main()  # ✅ Store returned results_df
    
    print("\n✅ Training complete! Results saved to:")
    print("- results/ directory")
    print("- model/ directory (saved model and scaler)")
    
    print("\nModel Performance Summary:")
    
    # # ✅ Fix sorting (since 'ROC-AUC' is not in results_df, sort by 'Accuracy' instead)
    # print(results_df.sort_values('Accuracy', ascending=False).to_markdown(index=False))  
    
    print("\nStarting prediction interface...")
    predict_new_patient()



# Load dataset
df = pd.read_csv('cleveland.csv', header=None)

# Rename columns
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

# Data Preprocessing
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})  # Convert target to binary (0/1)
df['sex'] = df.sex.map({0: 'female', 1: 'male'})  # Convert sex column
df['thal'] = df.thal.fillna(df.thal.mean())  # Fill missing values
df['ca'] = df.ca.fillna(df.ca.mean())  

# Convert categorical to numeric
df['sex'] = df.sex.map({'female': 0, 'male': 1}) 

# Splitting dataset
X = df.drop(columns=['target']).values  # All features except target
y = df['target'].values  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train SVM Model
classifier = SVC(kernel='rbf', probability=True)
classifier.fit(X_train, y_train)

# Save the model
joblib.dump(classifier, 'heart_disease_model.pkl')
joblib.dump(sc, 'scaler.pkl')

# Function for user input prediction
def predict_heart_disease():
    print("\nEnter patient details:")
    age = int(input("Age: "))
    sex = int(input("Sex (1 = Male, 0 = Female): "))
    cp = int(input("Chest Pain Type (0-3): "))
    trestbps = int(input("Resting Blood Pressure: "))
    chol = int(input("Cholesterol Level: "))
    fbs = int(input("Fasting Blood Sugar (1 = True, 0 = False): "))
    restecg = int(input("Resting ECG (0-2): "))
    thalach = int(input("Max Heart Rate: "))
    exang = int(input("Exercise-Induced Angina (1 = Yes, 0 = No): "))
    oldpeak = float(input("ST Depression Induced by Exercise: "))
    slope = int(input("Slope of Peak Exercise ST Segment (0-2): "))
    ca = int(input("Number of Major Vessels Colored by Fluoroscopy (0-3): "))
    thal = int(input("Thalassemia (1-3): "))

    # Create input array
    new_patient = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Load model and scaler
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Scale input data
    new_patient_scaled = scaler.transform(new_patient)

    # Predict
    prediction = model.predict(new_patient_scaled)

    # Display Result
    result = "Yes, the patient has heart disease." if prediction[0] == 1 else "No, the patient does NOT have heart disease."
    print("\n🩺 Prediction:", result)

# Run the prediction function
predict_heart_disease()
