# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, jsonify, request
import joblib


# Load data
data = pd.read_csv('./Data_2000_SA.csv')

# Separate features and target variable
X = data.drop(['Target-MCC1'], axis=1)
y = data['Target-MCC1']

# Label encode categorical features
cat_features = ['AgeGroup', 'MaritalStatus', 'Day', 'Gender', 'City']
le_dict = {}
for feature in cat_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])
    le_dict[feature] = le

    # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Make predictions on test data
y_pred = model.predict(X_test)


# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))


# Generate classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)


# Save the trained model and LabelEncoders
joblib.dump(model, 'model.joblib')
joblib.dump(le_dict, 'label_encoders.joblib')

# Load the trained model and LabelEncoders
model = joblib.load('model.joblib')
le_dict = joblib.load('label_encoders.joblib')

# Initialize Flask app
app = Flask(__name__)

# Define API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    print('request received')

    # Get data from request
    print('Here is data::::::::')
    data = request.get_json()
    print(data)

    print('Converting from JSON ...')
    data = pd.DataFrame.from_dict(data)
    print('Converted')

    print('encoding...')

    # # Label encode categorical features
    # for feature in cat_features:
    #     le = LabelEncoder()
    #     data[feature] = le.fit_transform(data[feature])

    # Encode new incoming data using the same LabelEncoders
    for feature in cat_features:
        le = le_dict[feature]
        data[feature] = le.transform([data[feature]])[0]

    print('incoming data encoded...')
    print(data)

    
    # Make prediction using trained model
    pred = model.predict(data)
    print(pred)

     # Make prediction using trained model
    proba = model.predict_proba(data)[0]
    pred_class = model.predict(data)[0]
    print(f'Prediction: {pred_class}')
    # Create dictionary to store class probabilities
    for i, p in enumerate(proba):
        print(f'Probability of class {model.classes_[i]}: {p:.4f}')

    # Create dictionary to store class probabilities
    prob_dict = {}
    for i, label in enumerate(model.classes_):
        prob_dict[label] = proba[i]


    # Sort probabilities in descending order and return top 3 predictions
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    # Return top 3 predictions as JSON
    return jsonify({'predictions': [{'class': str(label), 'probability': str(prob)} for label, prob in top3]})

    # # Return prediction as JSON
    # return jsonify({'prediction': (str(pred[0]))})

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True)
