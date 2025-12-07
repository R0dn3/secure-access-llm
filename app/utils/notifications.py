import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración SMTP para Outlook con contraseña de aplicación
SMTP_HOST = "smtp.office365.com"
SMTP_PORT = 587
SMTP_USER = "rdney1@hotmail.com"
SMTP_PASS = "ehhufecjrayeyowg"  # contraseña de aplicación
ALERT_EMAIL = "rdney1@hotmail.com"  # a quien enviar el reporte

def notify_email(subject: str, body: str) -> bool:
    """
    Envía un correo electrónico usando SMTP de Outlook con contraseña de aplicación.
    """
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())
        print("Correo enviado correctamente ✅")
        return True
    except smtplib.SMTPAuthenticationError as e:
        print(f"[❌ Error de autenticación]: {e}")
        return False
    except smtplib.SMTPConnectError as e:
        print(f"[❌ Error de conexión SMTP]: {e}")
        return False
    except Exception as e:
        print(f"[❌ Error general al enviar email]: {e}")
        return False
