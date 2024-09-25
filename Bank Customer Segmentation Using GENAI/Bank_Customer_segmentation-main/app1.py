import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

# Configure the API key
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("GENAI_API_KEY environment variable is not set.")
genai.configure(api_key=api_key)

# Load the pre-trained models and scalers
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('scaler1.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('knn2.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# Load the encoders
with open('encoder1.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)
expected_genders = {'Male', 'Female'}
expected_occupations = {'Salaried', 'Business', 'Housewife', 'Other'}
expected_education = {'Under Graduate', 'Graduate', 'Post Graduate'}
expected_married_status = {'yes', 'no'}
model_feature_columns = ['Age', 'Annual Income', 'AccountBalance', 'credit_score',
                         'Active_loans', 'Bank_Products', 'Fixed Deposit', 'Tenure',
                         'LastMonthTrans', 'Networth', 'Dependents',
                         'Occupation_Housewife', 'Occupation_Other', 'Occupation_Salaried',
                         'Gender_Male', 'Married_yes', 'education']


# Manually defined insights for each cluster
cluster_insights = {
    0: "Older customers with high annual income and substantial net worth, who have multiple active loans and are likely to be salaried or in business occupations.",
    1: "Middle-aged customers with moderate income and high account balances, possibly more conservative in spending, with fewer active loans.",
    2: "Younger customers or housewives with lower income and net worth, typically single with fewer active loans, and more likely to be in other or less stable occupations"
    #3: "Customers in this cluster prefer installment purchases and have moderate balances.",
    #4: "This cluster includes customers with high credit limits and occasional large purchases.",
    #5: "Customers in this cluster pay off their balances fully and regularly."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model', methods=['POST'])
def model():
    try:
        # Collecting features from the form
        features = {
            'Age': request.form.get('Age'),
            'Annual Income': request.form.get('Annual Income'),
            'Occupation': request.form.get('Occupation'),
            'AccountBalance': request.form.get('AccountBalance'),
            'credit_score': request.form.get('credit_score'),
            'Gender': request.form.get('Gender'),
            'Education': request.form.get('Education'),
            'Active_loans': request.form.get('Active_loans'),
            'Bank_Products': request.form.get('Bank_Products'),
            'Fixed Deposit': request.form.get('Fixed Deposit'),
            'Tenure': request.form.get('Tenure'),
            'LastMonthTrans': request.form.get('LastMonthTrans'),
            'Networth': request.form.get('Networth'),
            'Married': request.form.get('Married'),
            'Dependents': request.form.get('Dependents')
        }

        # Log received data for debugging
        logging.debug(f"Received features: {features}")

        # Validate and convert features to appropriate types
        for key in ['Age', 'Annual Income', 'AccountBalance', 'credit_score', 'Active_loans', 'Bank_Products', 'Tenure',
                    'LastMonthTrans', 'Networth', 'Dependents']:
            if features[key] is None or features[key] == '':
                return f"Missing or invalid value for {key}", 400
            try:
                features[key] = float(features[key])
            except ValueError:
                return f"Invalid value for {key}", 400

        # Convert 'Fixed Deposit' and 'Married' to binary
        features['Fixed Deposit'] = 1 if features['Fixed Deposit'] == 'Yes' else 0
        #features['Married'] = 1 if features['Married'] == 'yes' else 0

        # Validate categorical columns
        if features['Gender'] not in expected_genders:
            return "Invalid value for Gender", 400
        if features['Occupation'] not in expected_occupations:
            return "Invalid value for Occupation", 400
        if features['Education'] not in expected_education:
            return "Invalid value for Education", 400
        if features['Married'] not in expected_married_status:
            return "Invalid value for Married", 400

        # Convert features to DataFrame
        df_features = pd.DataFrame([features])

        # Process categorical features
        df_features['Occupation'] = df_features['Occupation'].astype(str)
        df_features['Gender'] = df_features['Gender'].astype(str)
        df_features['Education'] = df_features['Education'].astype(str)
        df_features['Married'] = df_features['Married'].astype(str)

        # Encode categorical features
        df_features_encoded = pd.DataFrame(onehot_encoder.transform(df_features[['Occupation', 'Gender','Married']]),
                                           columns=onehot_encoder.get_feature_names_out(['Occupation', 'Gender','Married']))

        # Add encoded columns to the DataFrame
        df_features_encoded = pd.concat([df_features.drop(columns=['Occupation', 'Gender','Married']), df_features_encoded],
                                        axis=1)

        # Encode 'Education' column using ordinal encoder
        df_features_encoded['education'] = ordinal_encoder.transform(df_features[['Education']])

        # Ensure the order of columns matches the model input
        for col in model_feature_columns:
            if col not in df_features_encoded.columns:
                df_features_encoded[col] = 0  # Add missing columns with default value

        df_features_encoded = df_features_encoded[model_feature_columns]

        # Scaling and PCA transformation
        scaled_data = scaler.transform(df_features_encoded)
        pca_transformed = pca.transform(scaled_data)
        Cluster = knn_model.predict(df_features_encoded)[0]

        # Get manual insight for the cluster
        insight = cluster_insights.get(Cluster, "No specific insight available for this cluster.")

        # Generate recommendations based on the insight using Gemini API
        recommendation = generate_recommendation(insight)

        return render_template('index.html',
                               prediction=f'This customer belongs to cluster {Cluster}',
                               insight=insight,
                               recommendation=recommendation,
                               cluster=Cluster)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return str(e), 500


@app.route('/generate_email', methods=['POST'])
def generate_email():
    try:
        cluster_str = request.form.get('cluster', '')

        if not cluster_str.isdigit():
            raise ValueError("Invalid cluster value")

        cluster = int(cluster_str)
        insight = cluster_insights.get(cluster, "No specific insight available for this cluster.")

        # Generate recommendations based on the cluster insight
        recommendations = generate_recommendation(insight)

        # Generate marketing email content based on recommendations
        subject, email_content = generate_marketing_email(recommendations)

        return render_template('index.html', email_content=email_content, subject=subject)
    except Exception as e:
        return str(e), 500

def generate_recommendation(insight):
    prompt = f"Based on the following insight: '{insight}', provide a list of 1 or 2 banking products in 2 lines. For example: loans, credit cards, savings accounts."

    recommendations = []

    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt, stream=True)
        for chunk in response:
            recommendations.append(chunk.text.strip())
    except Exception as e:
        print(f"Error during API call: {str(e)}")

    return "\n".join(recommendations)

def generate_marketing_email(recommendations):
    subject = "Exclusive Banking Products Tailored Just for You!"
    body = f"""
    Dear Valued Customer,

    We have personalized recommendations just for you based on your recent activity with us. Here are some products that we think you might find beneficial:

    {recommendations}

    Feel free to reach out to us for more details or to apply for any of these products. We're here to help you make the most of your financial opportunities!

    Best Regards,
    Your Bank's Marketing Team
    """

    return subject, body

if __name__ == '__main__':
    app.run(debug=True)
