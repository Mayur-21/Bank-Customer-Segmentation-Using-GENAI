import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
from flask import session

app = Flask(__name__)
app.config['SECRET_KEY'] = '2f246fb8e56c35ff90fafa91b5bb77de'

# Load environment variables from the .env file
load_dotenv()

# Configure the API key
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("GENAI_API_KEY environment variable is not set.")
genai.configure(api_key=api_key)

# Load the pre-trained models and scalers
with open('pca2.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('scaler3.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('rf1.2.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('encoder3.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('encoderord.pkl', 'rb') as f:
    ordinal_encoder = pickle.load(f)
smtp_server = os.getenv('SMTP_SERVER')
smtp_port = int(os.getenv('SMTP_PORT', 587))  # Default to port 587 if not set
smtp_user = os.getenv('SMTP_USER')
smtp_password = os.getenv('SMTP_PASSWORD')

# Ensure required SMTP settings are available
if not all([smtp_server, smtp_port, smtp_user, smtp_password]):
    raise ValueError("SMTP configuration environment variables are missing.")


# Define expected values for validation
expected_genders = {'Male', 'Female'}
expected_occupations = {'Salaried', 'Business', 'Housewife', 'Other'}
expected_education = {'Under Graduate', 'Graduate', 'Post Graduate'}
expected_married_status = {'yes', 'no'}
expected_vehicle = {'yes', 'no'}
expected_house = {'rented', 'owned'}
model_feature_columns = [
    'Age', 'Annual Income', 'AccountBalance', 'credit_score', 'Active_loans',
    'Bank_Products', 'Fixed Deposit', 'Tenure', 'LastMonthTrans', 'Networth',
    'Dependents', 'Vehicle', 'Occupation_Housewife', 'Occupation_Other',
    'Occupation_Salaried', 'Gender_Male', 'Married_yes', 'House_rented', 'education'
]

# Manually defined insights for each cluster
#cluster_insights = {
   # 0: f"Customers in Cluster 0 have  income and account balances. They have a high credit score and a decent net worth. This cluster has a significant number of salaried professionals and a balanced marital status. Vehicle ownership is common, and they have a moderate level of education.",

    #1: "Cluster 1 consists of customers with lower income and account balances compared to other clusters. They have a high credit score but a higher number of active loans. This cluster has a significant portion of housewives and other occupations, with a predominantly female gender distribution. Marital status is moderate.",

    #2: "Cluster 2 includes high-income customers with substantial account balances and net worth. They have a good credit score and are less involved in active loans. This cluster has the highest number of dependents and a significant proportion of married males. Almost 50% of people in this cluster own vehicles and have a moderate education level."
#}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/images')
def images():
    return render_template('Images.html')


csv_file_path = '/Users/harendrakshirsagar/Desktop/Cognizant Npn 2/data/Cluster_data.csv'


def save_to_csv(data, cluster):
    # Add the predicted cluster to the data
    data['Cluster'] = cluster
    data['Fixed Deposit'] = 'Yes' if data['Fixed Deposit'] == '1' or data['Fixed Deposit'] == 1 else 'No'
    data['Vehicle'] = 'yes' if data['Vehicle'] == '1' or data['Vehicle'] == 1 else 'no'

    # Check if the file exists
    file_exists = os.path.isfile(csv_file_path)

    # Define the CSV headers
    headers = list(data.keys())

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        # Write the headers if the file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(data)


@app.route('/model', methods=['POST'])
def model():
    # Collect form data
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
        'Dependents': request.form.get('Dependents'),
        'Vehicle': request.form.get('Vehicle'),
        'House': request.form.get('House')
    }

    # Validate and preprocess features
    for key in ['Age', 'Annual Income', 'AccountBalance', 'credit_score', 'Active_loans', 'Bank_Products', 'Tenure',
                'LastMonthTrans', 'Networth', 'Dependents']:
        if features[key] is None or features[key] == '':
            return f"Missing or invalid value for {key}", 400
        try:
            features[key] = float(features[key])
        except ValueError:
            return f"Invalid value for {key}", 400

    features['Fixed Deposit'] = 1 if features['Fixed Deposit'] == 'Yes' else 0
    features['Vehicle'] = 1 if features['Vehicle'].lower() == 'yes' else 0

    # Convert features to DataFrame
    df_features = pd.DataFrame([features])

    # Process categorical features
    df_features['Occupation'] = df_features['Occupation'].astype(str)
    df_features['Gender'] = df_features['Gender'].astype(str)
    df_features['Education'] = df_features['Education'].astype(str)
    df_features['Married'] = df_features['Married'].astype(str)
    df_features['Vehicle'] = df_features['Vehicle'].astype(str)
    df_features['House'] = df_features['House'].astype(str)

    # Encode categorical features using OneHotEncoder
    df_features_encoded = pd.DataFrame(
        onehot_encoder.transform(df_features[['Occupation', 'Gender', 'Married', 'House']]),
        columns=onehot_encoder.get_feature_names_out(
            ['Occupation', 'Gender', 'Married', 'House']))

    # Add encoded columns to the DataFrame
    df_features_encoded = pd.concat(
        [df_features.drop(columns=['Occupation', 'Gender', 'Married', 'House']), df_features_encoded],
        axis=1)

    # Encode 'Education' column using ordinal encoder
    df_features_encoded['education'] = ordinal_encoder.transform(df_features[['Education']])

    # Ensure the order of columns matches the model input
    for col in model_feature_columns:
        if col not in df_features_encoded.columns:
            df_features_encoded[col] = 0  # Add missing columns with default value

    df_features_encoded = df_features_encoded[model_feature_columns]

    # Scaling and PCA transformation (if applicable)
    scaled_data = scaler.transform(df_features_encoded)
    pca_transformed = pca.transform(scaled_data)

    # Predict cluster
    Cluster = model1.predict(df_features_encoded)[0]
    Cluster = int(Cluster)

    # Save customer data to CSV for logging
    save_to_csv(features, Cluster)
    vehicle_loan_recommendation = "We can recommend you consider a vehicle loan." if features[
                                                                                     'Vehicle'] == 'no' else "They already own a vehicle, so no need for a vehicle loan recommendation."
    home_loan_recommendation = "We can recommend you consider a home loan." if features[
                                                                               'House'] == 'rented' else "They already own home, so no need for a home loan recommendation."

    if features['Occupation'] == 'Business':
        occupation_recommendation = "We can recommend offering a business loan to help grow their business."
    elif features['Occupation'] == 'Other' and features['Age'] <= 22:
        occupation_recommendation = "We can recommend an education loan or a beginner credit card."
    elif features['Occupation'] == 'Housewife':
        occupation_recommendation = "We can suggest exploring a home renovation loan or investment products to secure their financial future."
    else:
        occupation_recommendation= "we can suggest offering them beginner level bank products"

    # Generate dynamic insight based on the cluster and form data
    if Cluster == 0:
        insight = (
            f"This customer has an annual income of {features['Annual Income']} and an account balance of {features['AccountBalance']}. "
            f"He have a credit score of {features['credit_score']} and a net worth of {features['Networth']}. "
            f"This cluster primarily includes salaried individuals with a balanced marital status. {vehicle_loan_recommendation} "
            f"{home_loan_recommendation} They have a moderate education level.")
        if features['Occupation'] == 'Other' and features['Age'] <=22:
            occupation_recommendation = " We can recommend an education loan or a beginner credit card."
            insight += occupation_recommendation
    elif Cluster == 1:
        insight = (
            f"This customer has an annual income of {features['Annual Income']} and an account balance of {features['AccountBalance']}. "
            f"This Customer have a credit score of {features['credit_score']} but also have {features['Active_loans']} active loans. "
            f"This cluster has a large proportion of housewives and other occupations. {occupation_recommendation}")

    elif Cluster == 2:
        insight = (
            f"Cluster 2 includes high-income customers. This customer has an annual income of {features['Annual Income']} and an account balance of {features['AccountBalance']}. "
            f"They have a credit score of {features['credit_score']}. Many in this cluster are married ,and they are key customers for the bank.")
    else:
        insight = "No specific insight is available for this cluster."

    # Generate recommendations based on the insight using Gemini API or similar
    recommendation = generate_recommendation(insight)
    session['insight'] = insight
    session['cluster'] = Cluster

    # Render the results on the web page
    return render_template('index.html',
                           prediction=f'This customer belongs to cluster {Cluster}',
                           insight=insight,
                           recommendation=recommendation,
                           cluster=Cluster)


@app.route('/generate_email', methods=['POST'])
def generate_email():

    insight = session.get('insight', "No specific insight available.")
    cluster = session.get('cluster', None)

    if cluster is None:
        return "Cluster information is missing.", 400

    # Generate recommendations based on the cluster insight
    recommendations = generate_recommendation(insight)


    # Generate marketing email content based on recommendations
    subject, email_content = generate_marketing_email(recommendations)

    return render_template('index.html', email_content=email_content, subject=subject, cluster=cluster)


def send_email_function(subject, email_content, recipient_email):
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(email_content, 'plain'))

    try:
        # Connect to the server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Upgrade the connection to secure
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipient_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


@app.route('/send_email', methods=['POST'])
def send_email():
    subject = request.form['subject']
    email_content = request.form['email_content']
    recipient_email = request.form['recipient_email']

    # Send the email
    try:
        send_email_function(subject, email_content, recipient_email)
        return redirect(url_for('index', message="Email sent successfully!"))
    except Exception as e:
        logging.error(f"Error occurred while sending email: {e}")
        return str(e), 500




def generate_recommendation(insight):
    prompt = f"Based on the following insight: '{insight}', provide a list of 1 or 2 banking products in 2-4 lines."

    recommendations = []
    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt, stream=True)
        for chunk in response:
            recommendations.append(chunk.text.strip())
    except Exception as e:
        print(f"Error during API call: {str(e)}")

    return "\n".join(recommendations)


def generate_marketing_email(recommendations):
    prompt = f"""
    Generate a marketing email in 8-10 lines based on the following recommendations:
    {recommendations}

    The email should be engaging and professional, without including any placeholder text like '[Customer Name]','[Bank Name] and any type of links and also do not write the subject. It should invite the customer to explore the recommended banking products and provide a call to action.
    """

    subject = "Exclusive Banking Products Tailored Just for You!"
    email_content = ""

    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt, stream=True)
        for chunk in response:
            email_content += chunk.text.strip() + "\n"
    except Exception as e:
        print(f"Error during API call: {str(e)}")

    return subject, email_content.strip()




if __name__ == '__main__':
    app.run(debug=True)
