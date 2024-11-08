import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

data = pd.read_csv('loan.csv')

# Drop the Loan_ID column
data = data.drop(columns=['Loan_ID'])

#Display basic information
print(data.info())

#Display the first few rows
print(data.head())

#Check for missing values
print(data.isnull().sum())

#Get summary statistics
print(data.describe())

#Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encoding categorical variables
encoder = LabelEncoder()
for column in data_filled.select_dtypes(include='object').columns:
    data_filled[column] = encoder.fit_transform(data_filled[column])

# Separate features and target variable
X = data_filled.drop('Loan_Status', axis=1)
y = data_filled['Loan_Status']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X_test)

# Feature importance
feature_importance = model.feature_importances_
feature_names = data_filled.columns.drop('Loan_Status')


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, '../Loan_App/model.pkl')

