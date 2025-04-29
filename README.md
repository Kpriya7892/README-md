# Project Name
Spam SMS Detection 
# introduction 
Spam detection is a critical task in the realm of text  classification , particularly in managing unsolicited messages in communication platforms. This project aims to develop a model that can accuretly classify SMS messages as spam or not spam using machine learning techniques.
# Dataset
The dataset used for this project is the UCI SMS Spam Collection, Which contains a collection of SMS messages labeled as either"ham" or "spam". The dataset is read into a pandaas DataFrame, ensuring that the encoding is set to 'latin-1' to handle any incompatible characters.
df = pd.read_csv('/content/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
# Data Preprocessing 
Data preprocessing is essential for improving the quality of the input data.
The following steps were taken:
Label Encoding: The labels were converted from categorical to numerical format. where "ham" is represented as 0 and "spam" as 1. Any NaN values in the label column were filled with -1.
Text Cleaning: A custom function was created to preprocess the text messages. This function converts text to lowercase, removes special characters and eliminates stopwords using the NLTK library.
# Feature Extraction
Text data was transformed using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This method converts text into numerical features that capture the importance of words relative to the dataset, reducing the influence of frequently occurring but less informative words.
# Model Training
The dataset was split into training and testing sets using an 80-20 split. A Multinomial Naive Bayes classifier was chosen for this task due to its effectiveness in text classification problems. The model was trained on the training set.
# Model Evaluation
After training the model, predictions were made on the test set. The performance of the model was evaluated using accuracy and a classification report, which includes precision, recall, and F1-score metrics. The accuracy score provides a quick overview of the model's performance.
# Code:
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load Dataset (Example uses UCI SMS Spam Collection)
# Specify the encoding as 'latin-1' to handle the incompatible characters
df = pd.read_csv('/content/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1') 

# 2. Encode Labels (ham = 0, spam = 1)
# Handle potential NaN values in the 'label' column before mapping
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1}).fillna(-1).astype(int) #convert to int & fill NaN with -1

# 3. Text Preprocessing Function
def preprocess(text):
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text) 
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
df['cleaned'] = df['message'].apply(preprocess)

# 4. Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label_num']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. Function to Predict if a Message is Spam or Not
def predict_spam(message):
    cleaned = preprocess(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return 'Spam' if prediction == 1 else 'Not Spam'

# 9. Prompt the User for Input Message
user_input = input("Enter an SMS message to check if it's spam: ")
result = predict_spam(user_input)
print("Prediction:", result)
# Output(Example 1) 
After training and evaluating the model:
Accuracy: 0.9811659192825112
              precision    recall  f1-score   support

           0       0.99      1.00      0.99       965
           1       0.95      0.87      0.91       150

    accuracy                           0.98      1115
   macro avg       0.97      0.93      0.95      1115
weighted avg       0.98      0.98      0.98      1115
# Then the script will prompt:
Enter an SMS message to check if it's spam
# If the user types
Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.
#Output
Prediction: Spam
# If the user types:
Hey, are we still on for lunch tomorrow?
# Output
Prediction: Not Spam
# Conclusion
The spam detection project successfully demonstrates the application of NIP and machine learning techniques to classify SMS messages. The model's performance metrics indicate its effectiveness in distinguishing between spam and non-spam messages. Future work may involve exploring more advanced models or incorporating additional features to enhance accuracy.


