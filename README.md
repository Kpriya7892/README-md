# Project Name
Spam SMS Detection 
# introduction 
Spam detection is a critical task in the realm of text  classification , particularly in managing unsolicited messages in communication platforms. This project aims to develop a model that can accuretly classify SMS messages as spam or not spam using machine learning techniques.
# Dataset
The dataset used for this project is the UCI SMS Spam Collection, Which contains a collection of SMS messages labeled as either"ham" or "spam". The dataset is read into a pandaas DataFrame, ensuring that the encoding is set to 'latin-1' to handle any incompatible characters.
df = pd.read_csv('/content/spam.csv', sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
# Data Preprocessing 



