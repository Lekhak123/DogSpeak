import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


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
    y_train = train_df['dog_id']
    X_train = train_df.drop(['file_name', 'path', 'split', 'dog_id'], axis=1)
    
    y_test = test_df['dog_id']
    X_test = test_df.drop(['file_name', 'path', 'split', 'dog_id'], axis=1)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)  # Only transform test data
    
    # Calculate class weights
    class_weights = {i: 1 / count for i, count in enumerate(np.bincount(y_train_encoded))}
    
    return (X_train, X_test, y_train_encoded, y_test_encoded, le.classes_, class_weights)

def cross_validate_and_evaluate(X, y, feature_type, classes, class_weights, n_splits=10):
    """Perform cross-validation and evaluate performance"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


    
    results = {}
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Initialize models with fixed parameters and class weights

        X_train_fold, X_test_fold = X.values[train_idx], X.values[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Impute missing values (mean strategy) before scaling
        imputer = SimpleImputer(strategy='mean')
        X_train_fold_imputed = imputer.fit_transform(X_train_fold)
        X_test_fold_imputed = imputer.transform(X_test_fold)
    
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold_imputed)
        X_test_fold_scaled = scaler.transform(X_test_fold_imputed)


        models = {
            'LR': LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                tol=1e-4,
                solver='lbfgs',
                random_state=RANDOM_STATE
            ),
            'RF': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                n_jobs=-1,
                random_state=RANDOM_STATE
            ),
            'XGB': xgb.XGBClassifier(
                max_depth=5,
                n_estimators=200,
                learning_rate=0.1,
                n_jobs=-1,
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                gpu_id=0,
                random_state=RANDOM_STATE
            )
        }

        for name, model in models.items():
            print(f"\nTraining {name} for {feature_type} (Fold {fold+1})...")
            # Initialize results dictionary for this model if it doesn't exist
            if name not in results:
                results[name] = []

            model.fit(X_train_fold_scaled, y_train_fold)
            y_pred = model.predict(X_test_fold_scaled)

            
            report = classification_report(y_test_fold, y_pred, output_dict=True)
            results[name].append(report)


            print(f"\n{name} Results (Fold {fold+1}):")
            print(classification_report(y_test_fold, y_pred))

            cm = confusion_matrix(y_test_fold, y_pred)
            plt.figure(figsize=(10,8))
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=classes,
                       yticklabels=classes)
            plt.title(f'Confusion Matrix - {name} ({feature_type} - Fold {fold+1})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f'results/{feature_type}/confusion_matrices/{name}_confusion_matrix_fold{fold+1}.png')
            plt.close()
    
    return results

def save_results(results, feature_type, classes):
    """Save results as CSV and LaTeX table with breed names, and calculate average metrics"""
    os.makedirs(f'results/{feature_type}', exist_ok=True)
    
    # Prepare results DataFrame
    metrics = []
    for model_name, result in results.items():
        fold_metrics = []
        for fold_idx, fold_result in enumerate(result):
            # Overall metrics
            overall = {
                'Model': f'{model_name} (Fold {fold_idx+1})',
                'Macro Avg F1': fold_result['macro avg']['f1-score'],
                'Macro Avg Precision': fold_result['macro avg']['precision'], 
                'Macro Avg Recall': fold_result['macro avg']['recall'],
                'Accuracy': fold_result['accuracy']
            }
            metrics.append(overall)
            fold_metrics.append(overall)
            
            # Per-class metrics
            for i, cls in enumerate(classes):
                metrics.append({
                    'Model': f'{model_name} ({cls} - Fold {fold_idx+1})',
                    'Macro Avg F1': fold_result[str(i)]['f1-score'],
                    'Macro Avg Precision': fold_result[str(i)]['precision'],
                    'Macro Avg Recall': fold_result[str(i)]['recall'],
                    'Accuracy': '-'  # No per-class accuracy
                })
        
        # Calculate average metrics across folds
        avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics if isinstance(fold[metric], (int, float))]) for metric in fold_metrics[0] if isinstance(fold_metrics[0][metric], (int, float))}
        avg_metrics['Model'] = f'{model_name} (Average)'
        metrics.append(avg_metrics)
    
    df_results = pd.DataFrame(metrics)
    
    # Save as CSV
    df_results.to_csv(f'results/{feature_type}/metrics.csv', index=False)
    
    # Generate LaTeX table
    latex_table = df_results.to_latex(index=False, float_format=lambda x: '%.3f' % x)
    with open(f'results/{feature_type}/metrics.tex', 'w') as f:
        f.write(latex_table)

def main():

    # path to the train_results.csv (Containing hubert embeddings)
    train_path = "./classification/hubert/dataset/train_results.csv"

    # path to the test_results.csv (Containing hubert embeddings)
    test_path = "./classification/hubert/dataset/test_results.csv"

    feature_type = "hubert"
    
    print(f"\nProcessing {feature_type} features...")
    
    # Setup directories
    setup_directories(feature_type)
    
    # Prepare data
    X_train, X_test, y_train, y_test, classes, class_weights = prepare_data(train_path, test_path)
    
    # Cross-validate and evaluate models
    results = cross_validate_and_evaluate(X_train, y_train, feature_type, classes, class_weights)
    
    # Save results
    save_results(results, feature_type, classes)

if __name__ == "__main__":
    main()