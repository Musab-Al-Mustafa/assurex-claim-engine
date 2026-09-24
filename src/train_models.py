import pandas as pd
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import ClaimPreprocessor

def train_and_evaluate():
    os.makedirs('model', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 1. Load Data Splits
    train_df = pd.read_csv('data/train/train_claims.csv')
    val_df = pd.read_csv('data/val/val_claims.csv')
    test_df = pd.read_csv('data/test/test_claims.csv')
    
    # 2. Preprocess Data
    preprocessor = ClaimPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    joblib.dump(preprocessor, 'model/preprocessor.joblib')
    
    # 3. Define Models to Train
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_val_acc = 0.0
    best_model_name = ""
    best_model = None
    
    report_text = "ASSUREX CLAIM ENGINE - DAY 2 MODEL EVALUATION REPORT\n"
    report_text += "=" * 60 + "\n\n"
    
    # 4. Train & Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        report_text += f"Model: {name}\n"
        report_text += f"  Train Accuracy: {train_acc * 100:.2f}%\n"
        report_text += f"  Val Accuracy:   {val_acc * 100:.2f}%\n"
        report_text += f"  Test Accuracy:  {test_acc * 100:.2f}%\n\n"
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_name = name
            best_model = model

    report_text += f"BEST PERFORMING MODEL: {best_model_name} (Val Acc: {best_val_acc * 100:.2f}%)\n"
    
    # 5. Detailed Report on Test Set for Best Model
    y_test_pred = best_model.predict(X_test)
    class_names = preprocessor.label_encoder.classes_
    report_text += "\nDetailed Test Set Classification Report:\n"
    report_text += classification_report(y_test, y_test_pred, target_names=class_names)
    
    # Save Report & Model
    with open('reports/tabular_model_report.txt', 'w') as f:
        f.write(report_text)
        
    joblib.dump(best_model, 'model/best_tabular_model.joblib')
    print(report_text)
    print("Training complete! Best model saved to model/best_tabular_model.joblib")

if __name__ == '__main__':
    train_and_evaluate()