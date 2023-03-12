# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from flask import Flask, jsonify, request
# import joblib

# data = pd.read_csv('./Data_2000_SA.csv')

# X = data.drop(['Target-MCC1', 'Target-MCC2', 'Target-MCC3'], axis=1)
# y_mcc1 = data['Target-MCC1']
# y_mcc2 = data['Target-MCC2']
# y_mcc3 = data['Target-MCC3']

# # Label encode categorical features
# cat_features = ['AgeGroup', 'MaritalStatus', 'Day', 'Gender', 'City']
# le_dict = {}
# for feature in cat_features:
#     le = LabelEncoder()
#     X[feature] = le.fit_transform(X[feature])
#     le_dict[feature] = le

# # Split data into training and testing sets
# X_train, X_test, y_train_mcc1, y_test_mcc1, y_train_mcc2, y_test_mcc2, y_train_mcc3, y_test_mcc3 = train_test_split(
#     X, y_mcc1, y_mcc2, y_mcc3, test_size=0.2, random_state=42)

# model_mcc1 = RandomForestClassifier(n_estimators=100, random_state=42)
# model_mcc1.fit(X_train, y_train_mcc1)

# model_mcc2 = RandomForestClassifier(n_estimators=100, random_state=42)
# model_mcc2.fit(X_train, y_train_mcc2)

# model_mcc3 = RandomForestClassifier(n_estimators=100, random_state=42)
# model_mcc3.fit(X_train, y_train_mcc3)

# # Save the trained models and LabelEncoders
# joblib.dump(model_mcc1, 'model_mcc1.joblib')
# joblib.dump(model_mcc2, 'model_mcc2.joblib')
# joblib.dump(model_mcc3, 'model_mcc3.joblib')
# joblib.dump(le_dict, 'label_encoders.joblib')

# # Load the trained models and LabelEncoders
# model_mcc1 = joblib.load('model_mcc1.joblib')
# model_mcc2 = joblib.load('model_mcc2.joblib')
# model_mcc3 = joblib.load('model_mcc3.joblib')
# le_dict = joblib.load('label_encoders.joblib')

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get data from request
#     data = request.get_json()
#     data = pd.DataFrame.from_dict(data)

#     # Encode new incoming data using the same LabelEncoders
#     for feature in cat_features:
#         le = le_dict[feature]
#         data[feature] = le.transform([data[feature]])[0]

#     pred_mcc1 = model_mcc1.predict_proba(data)
#     pred_mcc2 = model_mcc2.predict_proba(data)
   
