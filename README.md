# Flipkart Customer Satisfaction Prediction

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-0F766E)

An end-to-end machine learning project that analyzes Flipkart customer-support
interactions and predicts whether a ticket is at risk of receiving a low
customer satisfaction (CSAT) score.

## Business Objective

Customer satisfaction directly affects loyalty, retention, reviews, and brand
reputation. This project helps support teams:

- Identify the factors associated with low CSAT.
- Detect high-risk customer-support tickets early.
- Prioritize tickets that may require faster resolution or escalation.
- Improve agent training, shift planning, and support operations.

## Target Variable

The original `CSAT Score` is converted into a binary classification target
named `low_csat`:

- `1`: Low satisfaction, CSAT score from 1 to 3.
- `0`: Positive satisfaction, CSAT score from 4 to 5.

Because the target is imbalanced, model selection considers Precision, Recall,
F1 Score, Balanced Accuracy, and ROC-AUC instead of relying only on Accuracy.

## Project Workflow

1. Understand the dataset and business problem.
2. Inspect data types, missing values, and duplicate records.
3. Clean the data and convert date-time columns.
4. Engineer response-time and calendar features.
5. Analyze CSAT distribution and class imbalance.
6. Explore support channels, categories, shifts, tenure, and response time.
7. Encode categorical variables and scale numerical variables through a Scikit-learn pipeline.
8. Train Logistic Regression, Random Forest, and XGBoost classifiers.
9. Compare models using imbalance-aware evaluation metrics.
10. Select the best model and analyze feature importance.
11. Save the model and input options for Streamlit deployment.

## Model Selection

| Model | Purpose | Outcome |
|---|---|---|
| Logistic Regression | Interpretable baseline classifier | Selected as the best model based on F1 Score and Balanced Accuracy |
| Random Forest | Non-linear ensemble model | Competitive minority-class recall |
| XGBoost | Gradient-boosting model | High overall Accuracy but weaker low-CSAT Recall and F1 Score |

Logistic Regression was selected because identifying dissatisfied customers is
more important for this use case than maximizing overall Accuracy alone.

## Key Business Insights

- CSAT performance varies across support channels, issue categories, and sub-categories.
- Agent shift and tenure are useful indicators of service quality.
- Longer response times can increase the risk of low customer satisfaction.
- Accuracy alone can hide poor performance on dissatisfied customers.
- High-risk tickets should be prioritized for faster resolution and supervisor escalation.
- Feature-importance results can guide agent training and operational monitoring.

## Repository Structure

```text
Flipkart-Customer-Satisfaction-Prediction/
|-- README.md
|-- flipkart_customer_service_satisfaction.py
|-- Steps to run.txt
`-- Flipkart_App/
    |-- app.py
    |-- flipkart_low_csat_model.pkl
    `-- input_options.pkl
```

## Technology Stack

- Pandas and NumPy for data preparation.
- Matplotlib and Seaborn for exploratory data analysis.
- Scikit-learn for preprocessing, pipelines, modeling, and evaluation.
- XGBoost for boosted classification.
- Joblib for model serialization.
- Streamlit for the interactive prediction application.
- Google Colab for analysis and model training.

## Run the Streamlit Application

Clone the repository:

```bash
git clone https://github.com/ankit619288/Flipkart-Customer-Satisfaction-Prediction.git
cd Flipkart-Customer-Satisfaction-Prediction
```

Install the required packages:

```bash
py -m pip install streamlit pandas scikit-learn==1.8.0 joblib==1.5.3
```

Start the application from the folder containing the model files:

```bash
cd Flipkart_App
py -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

> Model files serialized with Joblib should be loaded with the same
> Scikit-learn and Joblib versions used during training.

## Run the Analysis

The analysis file was exported from Google Colab and contains Colab-specific
commands. To reproduce the training workflow:

1. Open the analysis in Google Colab.
2. Upload or mount `Customer_support_data.csv` in Google Drive.
3. Update the dataset path in the loading cell.
4. Run all cells in order.
5. Download the newly generated model and input-options files for deployment.

The dataset is not included in this repository.

## Streamlit Output

The application accepts ticket context, support-team details, and ticket
timeline inputs. It then displays:

- Predicted low-CSAT probability.
- High-risk or low-risk ticket status.
- Calculated response time.
- Recommended support actions.

## Conclusion

This project demonstrates how machine learning can support customer-service
operations by identifying low-CSAT risk before a customer becomes dissatisfied.
The resulting insights and Streamlit application can help teams improve response
time, escalation decisions, agent performance, customer loyalty, and retention.

## Author

**Ankit Mishra**  
[GitHub Profile](https://github.com/ankit619288)
