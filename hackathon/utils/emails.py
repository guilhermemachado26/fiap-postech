import os
import dotenv
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import cv2
import io


dotenv.load_dotenv()


def send_email(image_array):
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

    # Convert image array to a memory buffer
    _, buffer = cv2.imencode(".jpg", image_array)
    image_bytes = io.BytesIO(buffer.tobytes())

    # Attach image
    part = MIMEBase("application", "octet-stream")
    part.set_payload(image_bytes.getvalue())  # Read bytes from the buffer

    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename=detection.jpg")
    message.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        raise
