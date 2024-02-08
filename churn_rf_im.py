import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN

# Function to load data from file specified by the user
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

# Function to preprocess data
def preprocess_data(df):
    # Handling missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(how='any', inplace=True)
    
    # Feature engineering - binning tenure
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df['tenure_group'] = pd.cut(df['tenure'], range(1, 80, 12), right=False, labels=labels)
    
    # Dropping unnecessary columns
    df.drop(columns=['customerID', 'tenure'], axis=1, inplace=True)
    
    # Convert target variable to binary
    df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)
    
    # One-hot encoding categorical variables
    df_dummies = pd.get_dummies(df)
    
    return df_dummies

# Function to train and evaluate the model
def train_model(df):
    # Splitting data into features and target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Balancing the dataset using SMOTEENN
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    
    # Training Decision Tree Classifier
    model_dt = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=6, min_samples_leaf=8)
    model_dt.fit(X_resampled, y_resampled)
    
    # Training Random Forest Classifier
    model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
    model_rf.fit(X_resampled, y_resampled)
    
    # Saving Random Forest Classifier model
    filename = 'model_rf.sav'
    pickle.dump(model_rf, open(filename, 'wb'))
    pickle.dump(df.columns.tolist(), open('model_columns.sav', 'wb'))

    
    # Model evaluation
    dt_predictions = model_dt.predict(X_test)
    rf_predictions = model_rf.predict(X_test)
    
    dt_score = model_dt.score(X_test, y_test)
    rf_score = model_rf.score(X_test, y_test)

    print(f"Decision Tree Classifier Score: {dt_score}")
    print(f"Random Forest Classifier Score: {rf_score}")
    print("Decision Tree Classification Report:\n", classification_report(y_test, dt_predictions))
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

# Main function
if __name__ == "__main__":
    # Checking if file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_file>")
        sys.exit(1)
    
    # Loading data
    data_file = sys.argv[1]
    df = load_data(data_file)
    
    # Preprocessing data
    processed_df = preprocess_data(df)
    
    # Training and evaluating the model
    train_model(processed_df)
