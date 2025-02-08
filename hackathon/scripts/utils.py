import os
import dotenv
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


dotenv.load_dotenv()


def send_email():
    sender_email = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_EMAIL_PASSWORD")
    receiver_email = os.environ.get("RECEIVER_EMAIL")

    subject = "Alert -- Dangerous Object Detected"
    body = "This is a test email sent from FIAP"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        raise
