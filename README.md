# SMS Spam Detector Using Natural Language Processing (NLP)

## 1. Project Overview
The SMS Spam Detector is a machine learning project that classifies SMS messages as either SPAM or NOT SPAM.

The project uses Natural Language Processing (NLP) techniques and machine learning algorithms to identify unwanted or fraudulent text messages.

## 2. Project Objectives
- Develop a model that identifies spam messages.
- Apply text preprocessing techniques.
- Convert text messages into numerical features using TF-IDF.
- Train and compare machine learning models.
- Evaluate model performance.
- Build an interactive Streamlit application.

## 3. Dataset
The project uses the SMS Spam Collection Dataset from the UCI Machine Learning Repository.

The dataset contains SMS messages labelled as:
- Ham: legitimate messages.
- Spam: unwanted messages.

After removing duplicate records, 5,169 messages remained for model development.

Dataset source:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## 4. Technologies and Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- NLTK / Python string processing
- Joblib
- Streamlit

## 5. Project Workflow
1. Load the dataset.
2. Explore the data.
3. Check for missing values and duplicates.
4. Clean and preprocess SMS messages.
5. Encode the message labels.
6. Split the dataset into training and testing sets.
7. Convert text into numerical features using TF-IDF.
8. Train Naive Bayes and Logistic Regression models.
9. Evaluate and compare model performance.
10. Save the selected model and TF-IDF vectorizer.
11. Develop a Streamlit application.

## 6. Model Evaluation Results

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 94.87% | 100.00% | 59.54% | 74.64% |
| Logistic Regression | 96.03% | 95.92% | 71.76% | 82.10% |

Logistic Regression was selected because it achieved better overall performance and a better balance between spam precision and recall.

## 7. Sample Predictions
- "Congratulations! You have won a free prize. Call now!" — SPAM (92.66%)
- "Hello, how are you doing today?" — NOT SPAM (97.28%)
- "URGENT! Claim your cash reward now!" — SPAM (83.69%)

## 8. Streamlit Application
The application allows users to enter an SMS message and receive a prediction indicating whether the message is SPAM or NOT SPAM, together with a confidence score.

## 9. Project Structure
SMS-Spam-Detector/
- app.py
- requirements.txt
- README.md
- SMS_Spam_Detector.ipynb
- models/
  - sms_spam_detector.pkl

## 10. How to Run the Application

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## 11. Deployment
The application can be deployed using Streamlit Community Cloud by connecting the GitHub repository.

Deployment link: To be added after deployment.

## 12. Conclusion
This project demonstrates how Natural Language Processing and machine learning can be used to classify SMS messages. Logistic Regression achieved an accuracy of approximately 96.03% and was selected as the final model.
