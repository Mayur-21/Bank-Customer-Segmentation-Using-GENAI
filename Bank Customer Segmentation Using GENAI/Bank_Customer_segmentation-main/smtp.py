import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_function(subject, email_content, recipient_email):
    # Define SMTP server details
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587  # Use port 587 for TLS
    smtp_user = 'harendrakshirsagar38@gmail.com'  # Replace with your Gmail address
    smtp_password = 'toqr njkz irbr bvzl'  # Replace with your Gmail password or app password

    # Print debugging information
    print(f"SMTP Server: {smtp_server}")
    print(f"SMTP Port: {smtp_port}")

    # Create the email
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

# Example usage
send_email_function(
    subject="Test Subject",
    email_content="This is a test email.",
    recipient_email="recipient@example.com"
)
