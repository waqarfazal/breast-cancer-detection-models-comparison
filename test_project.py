#!/usr/bin/env python3
"""
Test script for Breast Cancer Detection Project
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

class TestBreastCancerDetection(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and models."""
        # Check if dataset exists
        self.assertTrue(os.path.exists('dataset.csv'), "Dataset file not found")
        
        # Load dataset
        self.dataset = pd.read_csv('dataset.csv')
        
        # Prepare data
        X = self.dataset[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']]
        y = self.dataset['Classification']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.05, random_state=42
        )
    
    def test_dataset_loading(self):
        """Test if dataset loads correctly."""
        self.assertIsNotNone(self.dataset)
        self.assertGreater(len(self.dataset), 0)
        self.assertIn('Classification', self.dataset.columns)
    
    def test_dataset_features(self):
        """Test if all required features are present."""
        required_features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']
        for feature in required_features:
            self.assertIn(feature, self.dataset.columns, f"Feature {feature} not found in dataset")
    
    def test_data_splitting(self):
        """Test if data splitting works correctly."""
        self.assertGreater(len(self.X_train), 0)
        self.assertGreater(len(self.X_test), 0)
        self.assertEqual(len(self.X_train), len(self.y_train))
        self.assertEqual(len(self.X_test), len(self.y_test))
    
    def test_logistic_regression(self):
        """Test Logistic Regression model."""
        model = LogisticRegression(solver='lbfgs', max_iter=4000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_decision_tree(self):
        """Test Decision Tree model."""
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
    
    def test_model_comparison(self):
        """Test that both models can be trained and compared."""
        # Train Logistic Regression
        lr_model = LogisticRegression(solver='lbfgs', max_iter=4000, random_state=42)
        lr_model.fit(self.X_train, self.y_train)
        lr_accuracy = accuracy_score(self.y_test, lr_model.predict(self.X_test))
        
        # Train Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(self.X_train, self.y_train)
        dt_accuracy = accuracy_score(self.y_test, dt_model.predict(self.X_test))
        
        # Both accuracies should be valid
        self.assertIsInstance(lr_accuracy, float)
        self.assertIsInstance(dt_accuracy, float)
        
        print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
        print(f"Decision Tree Accuracy: {dt_accuracy:.2%}")

def run_quick_test():
    """Run a quick test to verify the project works."""
    print("Running quick test...")
    
    try:
        # Import required modules
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        
        # Load data
        dataset = pd.read_csv('dataset.csv')
        print(f"✅ Dataset loaded successfully: {dataset.shape}")
        
        # Prepare data
        X = dataset[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']]
        y = dataset['Classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
        print(f"✅ Data prepared: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        
        # Test Logistic Regression
        lr_model = LogisticRegression(solver='lbfgs', max_iter=4000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
        print(f"✅ Logistic Regression trained: Accuracy = {lr_accuracy:.2%}")
        
        # Test Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
        print(f"✅ Decision Tree trained: Accuracy = {dt_accuracy:.2%}")
        
        print(f"\n🎉 All tests passed! Decision Tree {'outperforms' if dt_accuracy > lr_accuracy else 'underperforms'} Logistic Regression")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Breast Cancer Detection - Project Test")
    print("="*50)
    
    # Run quick test first
    if run_quick_test():
        print("\nRunning full test suite...")
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        print("Quick test failed. Please check your setup.")
