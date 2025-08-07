# Breast Cancer Detection using Machine Learning

## Overview

This project implements a machine learning-based approach for breast cancer detection using two different classification algorithms: **Decision Tree** and **Logistic Regression**. The study compares the performance of both models on the same dataset to determine which algorithm provides better classification accuracy.

## Key Findings

- **Decision Tree Classifier**: Achieved **100% accuracy** on the test set
- **Logistic Regression**: Achieved **50% accuracy** on the test set
- **Winner**: Decision Tree significantly outperformed Logistic Regression in this classification task

## Dataset

The project uses a breast cancer dataset containing the following features:
- **Age**: Patient's age
- **BMI**: Body Mass Index
- **Glucose**: Blood glucose levels
- **Insulin**: Insulin levels
- **HOMA**: Homeostatic Model Assessment
- **Leptin**: Leptin hormone levels
- **Adiponectin**: Adiponectin hormone levels
- **Resistin**: Resistin hormone levels
- **MCP.1**: Monocyte Chemoattractant Protein-1 levels
- **Classification**: Target variable (1 = Healthy, 2 = Breast Cancer)

**Dataset Statistics:**
- Total samples: 116
- Features: 9
- Classes: 2 (Healthy vs Breast Cancer)

## Project Structure

```
BreastCancerDetection/
├── BreastCancerDetection.ipynb    # Main Jupyter notebook with analysis
├── dataset.csv                    # Breast cancer dataset
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── .gitignore                     # Git ignore file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Open the notebook:**
   - Open `BreastCancerDetection.ipynb` in Jupyter

## Dependencies

The project requires the following Python packages:
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- jupyter

## Methodology

### Data Preprocessing
1. **Data Loading**: Loaded the dataset from CSV file
2. **Exploratory Data Analysis**: Analyzed data shape, columns, and null values
3. **Feature Selection**: Selected 9 relevant features for classification
4. **Data Splitting**: Split data into training (95%) and testing (5%) sets

### Model Training

#### 1. Logistic Regression
- **Algorithm**: Logistic Regression with LBFGS solver
- **Max Iterations**: 4000
- **Accuracy**: 50%
- **Confusion Matrix**: 
  ```
  [[3 2]
   [1 0]]
  ```

#### 2. Decision Tree
- **Algorithm**: Decision Tree Classifier
- **Random State**: 0 (for reproducibility)
- **Accuracy**: 100%
- **Confusion Matrix**:
  ```
  [[5 0]
   [0 1]]
  ```

## Results Analysis

### Performance Comparison

| Model | Accuracy | True Positives | True Negatives | False Positives | False Negatives |
|-------|----------|----------------|----------------|-----------------|-----------------|
| Decision Tree | 100% | 5 | 1 | 0 | 0 |
| Logistic Regression | 50% | 3 | 0 | 2 | 1 |

### Key Insights

1. **Decision Tree Superiority**: The Decision Tree classifier achieved perfect classification on the test set, correctly identifying all 6 test samples.

2. **Logistic Regression Limitations**: The linear model struggled with this dataset, achieving only 50% accuracy, suggesting the relationship between features and target may be non-linear.

3. **Feature Importance**: The Decision Tree's success indicates that the combination of multiple features (Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP.1) creates complex decision boundaries that are better captured by tree-based models.

## Usage

1. **Run the complete analysis:**
   - Execute all cells in `BreastCancerDetection.ipynb`
   - The notebook will automatically:
     - Load and explore the dataset
     - Train both models
     - Display accuracy scores and confusion matrices
     - Compare model performance

2. **Customize the analysis:**
   - Modify the test size in the train_test_split function
   - Adjust model parameters
   - Add additional evaluation metrics

## Future Improvements

1. **Cross-validation**: Implement k-fold cross-validation for more robust evaluation
2. **Feature Engineering**: Create additional features or perform feature selection
3. **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV
4. **Additional Models**: Test other algorithms like Random Forest, SVM, or Neural Networks
5. **Data Augmentation**: Increase dataset size for better generalization
6. **Model Interpretability**: Add feature importance analysis for Decision Tree

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a research project for educational purposes. The results should not be used for clinical decision-making without proper medical validation.
