import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

PATH = 'output'

# Load the dataset
data = pd.read_csv(os.path.join(PATH, 'dataset_merged_st.csv'))

# Drop the 'new_id' column
data = data.drop('new_id', axis=1)

data['datetime'] = pd.to_datetime(data['datetime'])

# Encode the 'id' column
data['id'] = data['id'].astype('category').cat.codes

# Split the dataset into features (X) and target (y)
X = data[['HR', 'EDA', 'TEMP']]
y = data['label']

# Impute missing values using the mean of the respective columns
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert the imputed data back to a DataFrame
X = pd.DataFrame(X_imputed, columns=X.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_clf.predict(X_test)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:", conf_matrix)
print("Classification Report:", class_report)

import pickle

with open('path/to/save/model.pkl', 'wb') as model_file:
    pickle.dump(rf_clf, model_file)

with open('path/to/save/model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Example: Make predictions using the loaded model
new_pred = loaded_model.predict(X_test)
