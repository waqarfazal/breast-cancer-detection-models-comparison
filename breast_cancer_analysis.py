#!/usr/bin/env python3
"""
Breast Cancer Detection using Machine Learning
==============================================

This script implements breast cancer detection using Decision Tree and Logistic Regression
algorithms. It compares the performance of both models on the same dataset.

Author: Waqar Fazal (www.waqarfazal.com)
Date: 2023
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_explore_data():
    """Load and explore the breast cancer dataset."""
    print("Loading dataset...")
    try:
        # Load the dataset
        dataset = pd.read_csv('dataset.csv')
        print(f"Dataset loaded successfully!")
        print(f"Shape: {dataset.shape}")
        print(f"Columns: {list(dataset.columns)}")
        
        # Check for null values
        print(f"\nNull values per column:")
        print(dataset.isna().sum())
        
        return dataset
    except FileNotFoundError:
        print("Error: dataset.csv not found in the current directory.")
        return None

def prepare_data(dataset):
    """Prepare the data for training."""
    print("\nPreparing data for training...")
    
    # Select features (all columns except Classification)
    X = dataset[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']]
    y = dataset['Classification']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression model."""
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*50)
    
    # Create and train the model
    log_model = LogisticRegression(solver='lbfgs', max_iter=4000, random_state=42)
    log_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = log_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Logistic Regression Accuracy: {accuracy:.2%}")
    print(f"Confusion Matrix:")
    print(cm)
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return log_model, y_pred, accuracy, cm

def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train and evaluate Decision Tree model."""
    print("\n" + "="*50)
    print("TRAINING DECISION TREE MODEL")
    print("="*50)
    
    # Create and train the model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Decision Tree Accuracy: {accuracy:.2%}")
    print(f"Confusion Matrix:")
    print(cm)
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return dt_model, y_pred, accuracy, cm

def compare_models(log_accuracy, dt_accuracy, log_cm, dt_cm):
    """Compare the performance of both models."""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    print(f"Logistic Regression Accuracy: {log_accuracy:.2%}")
    print(f"Decision Tree Accuracy: {dt_accuracy:.2%}")
    
    if dt_accuracy > log_accuracy:
        print(f"\n🎉 Decision Tree outperforms Logistic Regression by {dt_accuracy - log_accuracy:.2%}")
    else:
        print(f"\n🎉 Logistic Regression outperforms Decision Tree by {log_accuracy - dt_accuracy:.2%}")
    
    # Create comparison table
    comparison_data = {
        'Model': ['Logistic Regression', 'Decision Tree'],
        'Accuracy': [log_accuracy, dt_accuracy],
        'True Positives': [log_cm[1, 1], dt_cm[1, 1]],
        'True Negatives': [log_cm[0, 0], dt_cm[0, 0]],
        'False Positives': [log_cm[0, 1], dt_cm[0, 1]],
        'False Negatives': [log_cm[1, 0], dt_cm[1, 0]]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\nDetailed Comparison:")
    print(comparison_df.to_string(index=False))

def plot_results(log_cm, dt_cm):
    """Create visualization of the results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Logistic Regression confusion matrix
    sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Logistic Regression Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Plot Decision Tree confusion matrix
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Decision Tree Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrices saved as 'confusion_matrices.png'")
    plt.show()

def main():
    """Main function to run the complete analysis."""
    print("Breast Cancer Detection - Machine Learning Analysis")
    print("="*60)
    
    # Load and explore data
    dataset = load_and_explore_data()
    if dataset is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(dataset)
    
    # Train Logistic Regression
    log_model, log_pred, log_accuracy, log_cm = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    
    # Train Decision Tree
    dt_model, dt_pred, dt_accuracy, dt_cm = train_decision_tree(
        X_train, X_test, y_train, y_test
    )
    
    # Compare models
    compare_models(log_accuracy, dt_accuracy, log_cm, dt_cm)
    
    # Plot results
    try:
        plot_results(log_cm, dt_cm)
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
