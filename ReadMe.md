# IMDB Movie Reviews Sentiment Analysis

## Overview
This project aims to build a machine learning model capable of performing sentiment analysis on movie reviews from the IMDB dataset. By categorizing each review as either positive or negative, the model helps to derive insights into audience sentiments. This repository contains all the necessary code, datasets, and explanations to demonstrate the end-to-end process of data preparation, feature engineering, model building, and evaluation.

## Project Objectives
- Develop a machine learning model that can classify movie reviews as positive or negative.
- Use advanced data preprocessing and feature engineering techniques to prepare the dataset.
- Evaluate multiple models, including Logistic Regression, Naive Bayes, and CatBoost, to identify the best-performing approach.

## Dataset Description
The dataset used in this project is the IMDB movie reviews dataset, which contains 50,000 reviews labeled as either positive or negative. It is a balanced dataset, with an equal number of positive and negative sentiments, providing an unbiased foundation for training the model. The reviews are text-based, with varying length and complexity, making it a good challenge for developing robust NLP models.

## Project Workflow
1. **Data Understanding and Preprocessing**: The data was first loaded and explored to understand its characteristics. Key preprocessing steps included removing HTML tags, converting text to lowercase, removing outliers, and lemmatizing the text to create a clean dataset for analysis.

2. **Feature Engineering**: Text data was transformed using two main vectorization techniques—Bag of Words (BoW) and TF-IDF. Additionally, features such as review length, sentiment word count, and punctuation statistics were created to provide more context for model training.

3. **Modeling and Evaluation**: Several machine learning models were trained and evaluated, including Logistic Regression, Naive Bayes, and CatBoost Classifier. The models were assessed based on accuracy, precision, recall, and F1-score, with the CatBoost model showing the best performance.

4. **Hyperparameter Tuning**: The CatBoost model was further optimized using GridSearchCV to identify the best parameters, leading to improved accuracy and a more reliable model.

5. **Deployment**: The best-performing model was deployed using Streamlit to provide an easy-to-use interface for predicting the sentiment of movie reviews. Users can either input a single review or upload a CSV file for batch predictions.

## How to Run the Project
1. Clone this repository to your local machine.
   ```
   git clone https://github.com/gregorymikuro/IMDM-Reviews-Sentiment-Analysis-Project
   ```

2. Install the necessary dependencies.
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app to interact with the model.
   ```
   streamlit run main.py
   ```

4. You can input a single movie review or upload a CSV file to get sentiment predictions.

## Project Structure
- **data/**: Contains the dataset used for training and evaluation.
- **models/**: Stores the trained models and vectorizers.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and model experimentation.
- **main.py**: Streamlit app for model deployment.
- **README.md**: This file, providing an overview of the project.

## Requirements
- Python 3.7+
- Streamlit
- Pandas, Numpy
- Scikit-learn
- CatBoost
- spaCy
- BeautifulSoup

## Key Insights
- **Data Preprocessing**: Proper data cleaning and preprocessing are crucial to improve model performance. Techniques like removing HTML tags, stopwords, and lemmatization made a significant difference.
- **Model Comparison**: Logistic Regression and Naive Bayes provided solid baselines, but the CatBoost model outperformed them, especially after hyperparameter tuning.
- **Deployment**: The project highlights the importance of deploying models for real-world usage, making AI accessible through a simple web interface.

## Future Work
- **Deep Learning Models**: Explore LSTM or transformer-based models to potentially improve the sentiment analysis performance.
- **Additional Features**: Incorporate more advanced features such as word embeddings to capture the context better.
- **User Feedback Loop**: Implement a feedback mechanism to continuously improve the model based on user inputs.

## Conclusion
This project showcases a comprehensive approach to sentiment analysis, from data preprocessing to model deployment. By leveraging different machine learning models and techniques, we were able to build an effective sentiment classification system. The Streamlit app provides an accessible way to utilize the model, making it possible for users to quickly determine the sentiment of movie reviews.

## Contact
If you have any questions or suggestions, feel free to reach out to me.


- **LinkedIn**: [Gregory Mikuro](https://www.linkedin.com/in/gregorymikuro)

