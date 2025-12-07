# app/app.py

import streamlit as st
import subprocess
import os
import sys
import importlib.util
import pandas as pd

# ==========================
# CONFIGURACIÓN DE RUTAS
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DATASET_EMBEDDINGS = os.path.join(ROOT_DIR, "dataset_embeddings")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
CAMERA_ENROLL = os.path.join(ROOT_DIR, "camera", "camera_enroll.py")
CAMERA_VERIFY = os.path.join(ROOT_DIR, "camera", "camera_verify.py")
SEND_REPORT = os.path.join(ROOT_DIR, "app", "utils", "send_report.py")

# Crear carpetas si no existen
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(DATASET_EMBEDDINGS, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ==========================
# INTERFAZ STREAMLIT
# ==========================
st.set_page_config(
    page_title="Secure Access",
    page_icon="🔒",
    layout="wide"
)
st.title("🔒 Secure Access — Face Recognition + Anti-Spoofing")

tab1, tab2, tab3 = st.tabs(["📌 Enroll", "✅ Verify", "📄 Logs / Report"])

# ==========================
# TAB 1: REGISTRO DE USUARIO
# ==========================
with tab1:
    st.header("📌 Registro de usuario")
    st.info("Ingresa un ID, tu nombre y usa la cámara para registrar tu rostro.")

    user_id = st.text_input("🆔 User ID")
    user_name = st.text_input("👤 Nombre")

    if st.button("Iniciar Registro Facial"):
        if user_id and user_name:
            st.success("🚀 Abriendo la cámara para capturar tus imágenes...")
            subprocess.run([sys.executable, CAMERA_ENROLL, user_id, user_name])
        else:
            st.warning("⚠️ Por favor ingresa tu User ID y nombre.")

# ==========================
# TAB 2: VERIFICACIÓN FACIAL
# ==========================
with tab2:
    st.header("✅ Verificación facial en tiempo real")
    st.info("Coloca tu rostro frente a la cámara para autenticarte.")

    if st.button("Iniciar Verificación"):
        st.success("🚀 Abriendo la cámara para verificación facial...")
        subprocess.run([sys.executable, CAMERA_VERIFY])

# ==========================
# TAB 3: LOGS / REPORTES
# ==========================
with tab3:
    st.header("📄 Historial de accesos")
    log_path = os.path.join(LOGS_DIR, "access_logs.csv")

    if not os.path.exists(log_path):
        st.info("No hay registros disponibles.")
    else:
        try:
            df = pd.read_csv(log_path, header=0, quotechar='"', encoding="utf-8")
        except Exception as e:
            st.error(f"No se pudo leer el log: {e}")
            df = None

        if df is None or df.shape[0] <= 0:
            st.info("No hay registros aún.")
        else:
            while len(df.columns) < 4:
                df[df.shape[1]] = ""
            df = df.iloc[:, :4]
            df.columns = ["timestamp", "user_id", "status", "detail"]

            df["user_id"] = df["user_id"].astype(str).str.strip().replace(
                {"DESCONOCIDO": "desconocido", "None": "desconocido"}
            )
            df["status"] = df["status"].astype(str).str.strip()
            df["detail"] = df["detail"].astype(str).str.strip()

            st.markdown("**📊 Registros recientes:**")

            def color_row(row):
                s = str(row["status"]).upper()
                d = str(row["detail"]).lower()
                # 🟢 ACCESO VÁLIDO
                if s == "VALIDO":
                    return ['background-color: #d4edda; color: black'] * len(row)
                # 🔵 ATAQUE SPOOFING
                if s == "DENEGADO_SPOOF":
                    return ['background-color: #e3f2fd; color: black'] * len(row)
                # 🟡 FALLO DE PATRÓN
                if s == "DENEGADO_PATRON" or "pattern=" in d:
                    return ['background-color: #fff3cd; color: black'] * len(row)
                # 🔴 DENEGADO NORMAL
                if s == "DENEGADO":
                    return ['background-color: #f8d7da; color: black'] * len(row)
                # ⚪ OTROS
                return ['background-color: white; color: black'] * len(row)
            

            styled = df.style.apply(color_row, axis=1).set_properties(
                **{'text-align': 'left'}
            ).set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#f0f2f6'),
                           ('color', 'black'),
                           ('text-align', 'left')] }
            ])
            st.dataframe(styled, use_container_width=True, height=360)

            st.markdown("---")
            st.subheader("🧠 Generar reporte inteligente")

            if st.button("Generar reporte con Llama3"):
                st.info("Analizando registros con IA local...")
                try:
                    # Cargar módulo LLM
                    llm_path = os.path.join(ROOT_DIR, "app", "utils", "llm.py")
                    spec = importlib.util.spec_from_file_location("llm", llm_path)
                    llm = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(llm)

                    logs_dicts = df.to_dict(orient="records")
                    report = llm.summarize_logs_as_report(logs_dicts)

                    # Mostrar reporte en Streamlit
                    st.success("✅ Reporte generado correctamente:")
                    st.text_area("📋 Reporte IA (resultado):", report, height=340)

                    # --------------------------
                    # Enviar automáticamente con send_report.py
                    # --------------------------
                    st.info("📧 Enviando reporte por correo...")
                    result = subprocess.run(
                        [sys.executable, SEND_REPORT, report],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("✅ El reporte fue enviado correctamente a tu correo.")
                    else:
                        st.error(f"❌ Error al enviar correo: {result.stderr}")

                except Exception as e:
                    st.error(f"❌ Error al generar reporte: {e}")
