# send_report.py

import sys
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración SMTP (Gmail)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "waberto113@gmail.com"
SMTP_PASS = # Contraseña de aplicación preferible gmail
ALERT_EMAIL = # A quién enviar el reporte

def send_email(report_text):
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL
    msg["Subject"] = "Reporte Inteligente de Accesos"
    msg.attach(MIMEText(report_text, "plain", "utf-8"))

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())
        print("Correo enviado correctamente")  # quitamos emojis
        return True
    except Exception as e:
        print(f"[Error al enviar correo]: {e}")  # quitamos emojis
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python send_report.py 'Texto del reporte aquí'")
        sys.exit(1)

    # Soportar reportes grandes
    report = sys.argv[1]
    send_email(report)
