import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import BertTokenizer, BertModel

# Load the training and test datasets
train_df = pd.read_csv(r'C:\Users\desti\OneDrive - JK LAKSHMIPAT UNIVERSITY\train.csv')
test_df = pd.read_csv(r'C:\Users\desti\OneDrive - JK LAKSHMIPAT UNIVERSITY\test.csv')

# Define the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define the training data
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

# Convert the text data to numerical features using BERT
train_features = []
for text in train_texts:
    encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    features = output[0].mean(dim=1).squeeze().tolist()
    train_features.append(features)

# Define the test data
test_texts = test_df['text'].tolist()

# Convert the text data to numerical features using BERT
test_features = []
for text in test_texts:
    encoded_text = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    features = output[0].mean(dim=1).squeeze().tolist()
    test_features.append(features)

# Train an SVM model on the training data
svm_model = svm.SVC(kernel='linear')
svm_model.fit(train_features, train_labels)

# Use the trained model to predict stress levels for the test data
predicted_labels = svm_model.predict(test_features)

# Print the predicted stress levels for each test text
for i in range(len(test_texts)):
    print('Text: ' + test_texts[i] + '\tPredicted stress level: ' + str(predicted_labels[i]))
