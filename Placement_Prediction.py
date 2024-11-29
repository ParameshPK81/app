import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('data2.csv')

# Drop unnecessary columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Handle missing values
df.fillna(0, inplace=True)

# Feature-target split
x = df.drop(['StudentId', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Encode categorical variables
le = preprocessing.LabelEncoder()
x['Internship'] = le.fit_transform(x['Internship'])
x['Hackathon'] = le.fit_transform(x['Hackathon'])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# Train the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
classifier.fit(x_train, y_train)

# Test the model
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model using pickle
pickle.dump(classifier, open('model.pkl', 'wb'))

# Load the saved model for predictions
model = pickle.load(open('model.pkl', 'rb'))

# Example prediction (adjust input values as needed)
sample_input = [[8, 1, 3, 2, 9, 4.8, 0, 1, 71, 87, 0]]
result = model.predict(sample_input)
print(f"Prediction for input {sample_input}: {result}")
