# Instagram-Account-Prediction
# README: Machine Learning Project

## Overview
This repository contains the implementation and analysis of a machine learning project focused on predictive modeling. The main objective is to develop models to predict social media engagement metrics, with a specific emphasis on like counts. Key sections of the project include:

1. **Data Preparation**: Scripts for processing input JSONL files and structuring training and testing datasets.
2. **Model Training and Evaluation**: Functions to train predictive models and compute evaluation metrics.
3. **Prediction Pipeline**: Methods for generating predictions and processing outputs.
4. **Results Analysis**: Metrics and visualizations to compare model performance on training and test datasets.
5. **Model Deployment**: Suggestions for deploying the model in a real-world scenario.

**Key Files:**
- `process_jsonl_predictions.py`: Main script for data processing and prediction.
- `412_project_selim.ipynb`: Jupyter Notebook for exploratory data analysis and model evaluation.

## Methodology
The project follows a systematic approach to solve the problem:

1. **Data Processing:**
   - Input JSONL files are read and parsed.
   - Posts are grouped by username, and relevant features such as like counts and timestamps are extracted.
   - Missing or incomplete data is handled through default values and preprocessing techniques.

2. **Modeling:**
   - A time-weighted approach is used to predict like counts. This considers the logarithmic transformation of like counts to handle skewness.
   - The `log_mse_like_counts` function evaluates model performance by calculating the log-transformed mean squared error.

3. **Prediction Pipeline:**
   - Posts from training and test datasets are processed to generate predictions.
   - Outputs are saved as JSONL files with predicted like counts added to each entry.
   - The pipeline includes robust error handling to skip problematic data entries while logging issues.

4. **Evaluation:**
   - Metrics such as Log Mean Squared Error (Log MSE) are computed for both training and test datasets.
   - Summary statistics of the predicted values are calculated, including mean, median, min, and max values.
   - Comparative analysis is performed to identify areas of improvement.

5. **Deployment Considerations:**
   - Suggestions for deploying the model in a production environment, such as API integration or batch processing workflows.
   - Recommendations for continuous monitoring and retraining based on new data.

## Results
### Model Performance Metrics
The model's performance was assessed using the Log MSE metric. The results are as follows:

| Dataset         | Log MSE   | Samples Processed |
|-----------------|-----------|-------------------|
| Training Set    | 1.096093  | 94,824            |
| Test Set        | 0.712760  | 92,478            |

Additionally, accuracy metrics for the models were computed to evaluate classification-like performance:

| Metric                                    | Value             |
|------------------------------------------|-------------------|
| Training Accuracy (Feature Selection)    | 99.68%            |
| Validation Accuracy (Feature Selection)  | 65.39%            |
| Cross-validation Mean Accuracy           | 73.08%            |

### Detailed Classification Metrics
#### Training Classification Report (Feature Selection):
```
                      precision    recall  f1-score   support

                 art       1.00      0.99      1.00       153
       entertainment       1.00      1.00      1.00       258
             fashion       1.00      1.00      1.00       239
                food       0.98      1.00      0.99       409
              gaming       1.00      1.00      1.00        10
health and lifestyle       1.00      0.99      1.00       402
    mom and children       1.00      1.00      1.00       119
              sports       1.00      1.00      1.00        90
                tech       1.00      1.00      1.00       277
              travel       1.00      1.00      1.00       235

            accuracy                           1.00      2192
           macro avg       1.00      1.00      1.00      2192
        weighted avg       1.00      1.00      1.00      2192
```

### Prediction Statistics
Predicted like counts were analyzed to summarize key metrics:

| Statistic       | Value      |
|-----------------|------------|
| Mean Predicted  | 6180.49    |
| Median Predicted| 75.50      |
| Min Predicted   | 0          |
| Max Predicted   | 985,410    |

### Sample Predictions
Below are examples of predictions for social media posts:

#### Example 1:
- **Username**: `kozayarismasi`
- **Original Like Count**: 170
- **Predicted Like Count**: 160 (rounded)
- **Caption**: "KOZA 2023 2.si Damla’nın koleksiyonu..."

#### Example 2:
- **Username**: `celikbeymobilya`
- **Original Like Count**: 122
- **Predicted Like Count**: 130 (rounded)
- **Caption**: "Tüm Türkiye ve Avrupa’ya sevkiyatlarımız..."

#### Example 3:
- **Username**: `girisimci_muhendis`
- **Original Like Count**: 1251
- **Predicted Like Count**: 1200 (rounded)
- **Caption**: "Daha Fazlası İçin Beğenmeyi ve Takip Etmeyi Unutmayın..."

## Experimental Findings
### Key Observations
- The model demonstrated lower error on the test set (Log MSE = 0.712760) compared to the training set (Log MSE = 1.096093), indicating effective generalization.
- Predictions align closely with actual values for median-like posts, while higher deviation is observed for outliers with exceptionally high like counts.
- The logarithmic transformation effectively reduced the impact of outliers, making the predictions more reliable.
- Accuracy metrics for training were remarkably high (99.68%), though validation accuracy was moderate (65.39%), suggesting potential overfitting.
- Cross-validation yielded a mean accuracy of 73.08%, indicating reasonable generalization capability across folds.

### Improvements and Future Work
- **Feature Engineering**: Incorporating additional features such as hashtags, media type, and comment counts may improve model performance.
- **Advanced Models**: Exploring more complex machine learning algorithms or neural networks could reduce prediction errors further.
- **Dataset Expansion**: Using a larger and more diverse dataset would enhance model robustness.
- **Hyperparameter Tuning**: Further optimization of model parameters could yield better results.

## Instructions to Run the Code
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the prediction pipeline:
   ```bash
   python process_jsonl_predictions.py
   ```
4. Review the output predictions in the specified JSONL output file.
5. Use the provided Jupyter Notebook for additional analysis and visualization.

## Acknowledgments
This project was developed as part of a machine learning homework assignment, showcasing skills in data processing, predictive modeling, and results analysis.

Special thanks to the course instructors and peers for their valuable guidance and feedback. The dataset used in this project was obtained from publicly available social media data, ensuring compliance with privacy and ethical standards.

---

For any questions or suggestions, please feel free to open an issue or contact the repository maintainer.


