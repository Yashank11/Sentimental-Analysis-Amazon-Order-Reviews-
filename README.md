Sentimental Analysis Model (Based on Amazon Order Reviews)

Hello Everyone! Myself Yashank Rajvanshi, an enthusiast Data Analytics Engineer with an experience of more than 4 years in this field and having knowlegde of Python, PowerBI, DAX, VBA, Advanced Excel, Artificial 
Intelligence, Preprocessing Tools, Machine Learning. I am currently in my Final of year B.Tech Degree. My dicipline is Information Technology (IT). I have completed various projects and a 5 month internship experience 
as Data Analytics Intern at Kashsam Data Solutions. 

To access dataset, follow the below link:
https://drive.google.com/file/d/1IVQwaEmF1KoctCofaG7LGteV0XJ6lgiY/view?usp=sharing

Process Overview of the Sentiment Analysis Code Execution
1. Library Imports:
The necessary libraries for data processing, visualization, machine learning, and model evaluation are imported. These include:

pandas for data manipulation.
matplotlib for plotting.
sklearn for machine learning models and metrics.
nltk for text processing and Naive Bayes classification.
numpy for numerical operations.
re and string for regular expression-based text cleaning.
wordcloud for creating word clouds.
shap for model interpretability.
2. Data Loading:
The dataset is loaded using pandas.read_csv() and the relevant columns (reviews.rating, reviews.text, reviews.title, reviews.username) are extracted for analysis.

3. Handling Missing Values:
Checking for Null Values: A check is done to see if any null values exist in the dataset.
Filtering Null and Not Null Values: Reviews with missing ratings are separated from those with valid ratings.
4. Labeling Sentiments:
The reviews are classified into positive (reviews.rating >= 4) and negative categories based on the rating.
A bar plot visualizes the count of positive and negative reviews.
5. Text Preprocessing:
Text Cleaning: Reviews are cleaned using a regular expression to remove non-alphabetic characters and convert all text to lowercase.
Tokenization: Words are tokenized (split) for further processing.
6. Data Splitting:
The dataset is split into training (80%) and testing (20%) sets to train and evaluate the models.

7. Feature Extraction for Naive Bayes:
A feature extractor (word_feats) is created to represent each word in a review as a feature.
Training and Testing Data Preparation: The training and test data are converted into a format suitable for the Naive Bayes classifier.
8. Naive Bayes Classifier:
The NLTK Naive Bayes classifier is trained on the training data.
The accuracy of the model is calculated, and the top informative features are displayed.
9. TF-IDF and CountVectorizer:
CountVectorizer: Converts the cleaned text data into a sparse matrix of word counts.
TF-IDF Transformation: The word counts are transformed into TF-IDF scores for each review.
This transformation is applied to both the training and test data, as well as reviews with missing ratings (for future prediction).
10. Multinomial Naive Bayes Model:
A Multinomial Naive Bayes classifier is trained on the TF-IDF features of the training data.
The model is evaluated on the test data, and predictions are made for reviews with missing ratings.
11. Bernoulli Naive Bayes Model:
A Bernoulli Naive Bayes classifier is trained similarly and evaluated using the same process as the Multinomial model.
12. Logistic Regression:
A Logistic Regression model is trained on the TF-IDF features.
The accuracy is calculated, and predictions are made for both the test data and the reviews with missing ratings.
13. Feature Importance from Logistic Regression:
The top positive and negative features are extracted from the Logistic Regression model, showing which words contribute the most to positive and negative predictions.
14. ROC Curve Comparison:
The ROC curves of the Multinomial Naive Bayes, Bernoulli Naive Bayes, and Logistic Regression models are plotted to compare the performance of the classifiers.
15. Precision-Recall Curve:
A Precision-Recall curve is plotted to visualize the trade-off between precision and recall for each classifier.
16. Sentiment Distribution:
A pie chart is created to visualize the distribution of positive and negative sentiments in the dataset.
17. Word Cloud Visualization:
Word clouds are generated for both the entire dataset and for positive and negative reviews separately, to show the most common words used.
18. Learning Curve:
A learning curve is plotted to show the training and validation scores as a function of the number of training examples, which helps visualize if the model is overfitting or underfitting.
19. Model Calibration Curves:
Calibration curves are plotted for each model to assess how well the predicted probabilities align with the true probabilities of the classes.
20. Class Imbalance Analysis:
Bar charts are used to analyze the distribution of sentiments in both the training and test sets.
21. Misclassification Analysis:
Misclassified examples (where the predicted sentiment doesn’t match the actual sentiment) from the test set are identified and displayed.
22. Shapley Values for Feature Importance:
SHAP (Shapley Additive Explanations) is used to explain the output of the Logistic Regression model, helping understand how each feature influences the predictions.
23. Error Analysis: Precision and Recall for Each Model:
The classification report (precision, recall, and F1-score) is calculated for each classifier, giving a detailed evaluation of model performance.
Execution Steps:
Ensure Data File Availability: Place the 1429_1.csv dataset in the correct directory.
Install Necessary Libraries: Ensure all required libraries (nltk, sklearn, shap, matplotlib, etc.) are installed.
Run the Script: Execute the script in a Jupyter notebook or any Python IDE.
Evaluate Outputs: Observe and interpret the output of each section, especially the model performance metrics, visualizations, and feature importance
