import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from cuml.svm import SVC as cuSVC  # GPU-accelerated SVM
    gpu_svm_available = True
except ImportError:
    gpu_svm_available = False

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def setup_directories(feature_type):
    """Create necessary directories"""
    dirs = [
        f'results/{feature_type}',
        f'results/{feature_type}/confusion_matrices',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def prepare_data(train_path, test_path):
    """Prepare data with stratified split and print class distributions"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features from metadata
    y_train = train_df['gender']
    X_train = train_df.drop(['file_name', 'path', 'split', 'gender'], axis=1)
    
    y_test = test_df['gender']
    X_test = test_df.drop(['file_name', 'path', 'split', 'gender'], axis=1)
    

    # Print initial NaN statistics
    print("\nMissing values before handling:")
    print("Train set NaN count:", X_train.isna().sum().sum())
    print("Test set NaN count:", X_test.isna().sum().sum())


    # Fill NaN values with the mean of each column
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())  # Ensure test has same treatment

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.fit_transform(y_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Feature dimensions: {X_train_scaled.shape[1]}")

    # Print class distributions
    print("\nClass Distribution:")
    print("Train set:")
    for cls in le.classes_:
        count = sum(y_train == cls)
        print(f"{cls}: {count} samples")
    
    print("\nTest set:")
    for cls in le.classes_:
        count = sum(y_test == cls)
        print(f"{cls}: {count} samples")
    
    return (X_train_scaled, X_test_scaled, 
            y_train_encoded, y_test_encoded, 
            le.classes_)


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_type, classes):
    """Train models and evaluate performance"""
    
    # Initialize models with fixed parameters
    models = {
        'LR': LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            tol=1e-4,
            solver='lbfgs'
        ),
        'RF': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            n_jobs=-1
        ),
        'XGB': xgb.XGBClassifier(
            max_depth=5,
            n_estimators=200,
            learning_rate=0.1,
            n_jobs=-1,
            tree_method='hist',    # Use standard histogram algorithm
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} for {feature_type}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Get metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = report
        
        # Print results
        print(f"\n{name} Results:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=classes,
                   yticklabels=classes)
        plt.title(f'Confusion Matrix - {name} ({feature_type})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'results/{feature_type}/confusion_matrices/{name}_confusion_matrix.png')
        plt.close()
    
    return results

def save_results(results, feature_type, classes):
    """Save results as CSV and LaTeX table with gender names"""
    os.makedirs(f'results/{feature_type}', exist_ok=True)
    
    # Prepare results DataFrame
    metrics = []
    for model_name, result in results.items():
        # Overall metrics
        overall = {
            'Model': model_name,
            'Macro Avg F1': result['macro avg']['f1-score'],
            'Macro Avg Precision': result['macro avg']['precision'], 
            'Macro Avg Recall': result['macro avg']['recall'],
            'Accuracy': result['accuracy']
        }
        metrics.append(overall)
        
        # Per-class metrics
        for i, cls in enumerate(classes):
            metrics.append({
                'Model': f'{model_name} ({cls})',
                'Macro Avg F1': result[str(i)]['f1-score'],
                'Macro Avg Precision': result[str(i)]['precision'],
                'Macro Avg Recall': result[str(i)]['recall'],
                'Accuracy': '-'  # No per-class accuracy
            })
    
    df_results = pd.DataFrame(metrics)
    
    # Save as CSV
    df_results.to_csv(f'results/{feature_type}/metrics.csv', index=False)
    
    # Generate LaTeX table
    latex_table = df_results.to_latex(index=False, float_format=lambda x: '%.3f' % x)
    with open(f'results/{feature_type}/metrics.tex', 'w') as f:
        f.write(latex_table)


def main():
    
    # Base path containing the subfolders acoustic features
    base_path = "./classification/acoustic/dataset"
    
    # Define feature types to process
    feature_types = ['mfcc', 'gemaps', 'filterbank', 'egemaps']
    for feature_type in feature_types:
        train_path = f"{base_path}/{feature_type}/train/results.csv"
        test_path = f"{base_path}/{feature_type}/test/results.csv"
        print(f"\nProcessing {feature_type} features...")
        
        # Setup directories
        setup_directories(feature_type)
        
        # Prepare data
        X_train, X_test, y_train, y_test, classes = prepare_data(train_path, test_path)
        
        # Train and evaluate models
        results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_type, classes)
        
        # Save results
        save_results(results, feature_type, classes)

if __name__ == "__main__":
    main()
