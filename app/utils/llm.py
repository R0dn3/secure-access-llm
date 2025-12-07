# llm.py
import json
import requests
from datetime import datetime, timezone, timedelta
import re
import math

# URL correcta del endpoint de Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:latest"

def summarize_logs_as_report(logs, block_size=50):
    """
    Genera un informe de seguridad en español usando Ollama (Llama 3),
    con formato narrativo y usuarios entre comillas.
    Funciona con cualquier cantidad de logs dividiéndolos en bloques.
    """
    try:
        if not logs:
            return "No hay registros para generar el reporte."

        # Normalizar logs
        logs_local = []
        for log in logs:
            new_log = log.copy()

            # Convertir hora a local (Perú UTC-5)
            if "ts" in new_log:
                try:
                    dt = datetime.fromisoformat(new_log["ts"].replace("Z", ""))
                    dt_local = dt.astimezone(timezone(timedelta(hours=-5)))
                    new_log["ts"] = dt_local.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

            # Normalizar user_id
            if "user_id" in new_log:
                uid = str(new_log["user_id"]).strip()
                if uid.lower() == "desconocido":
                    new_log["user_id"] = "desconocido"
                else:
                    new_log["user_id"] = str(new_log["user_id"])

            # Normalizar status/result
            if "status" in new_log:
                status = str(new_log["status"]).upper()
                status = status.replace("(", "").replace(")", "").replace("_", " ")
                new_log["status"] = status.title()

            # Si hay "pattern=XXX", convertir a JSON puro
            if "pattern" in new_log:
                new_log["pattern"] = str(new_log["pattern"]).upper()

            logs_local.append(new_log)

        # Dividir logs en bloques
        total_blocks = math.ceil(len(logs_local) / block_size)
        final_report = ""

        for i in range(total_blocks):
            block_logs = logs_local[i*block_size:(i+1)*block_size]
            try:
                logs_text = json.dumps(block_logs, indent=2, ensure_ascii=False)

                prompt = f"""
Eres un analista de ciberseguridad experto. Responde únicamente en español 
y redacta un informe profesional, claro y bien estructurado con el siguiente formato:

**Análisis de eventos de acceso**
Desarrolla un párrafo explicativo sobre las tendencias observadas en los accesos. 
Menciona explícitamente los usuarios entre comillas, por ejemplo: "1" o "desconocido". 
No uses paréntesis con términos técnicos. 
Usa un tono formal y técnico.
Cabe mencionar que preferiblemente no uses "pattern=FAIL" u otros en esos casos interpretalos
Y no te olvides de usar (hora local, usuario, resultado, score, spoof).

**Comentarios**
Describe hallazgos relevantes o posibles anomalías observadas. Evita paréntesis.
Cabe mencionar que preferiblemente no uses "pattern=FAIL" u otros en esos casos interpretalos
Y no te olvides de usar (hora local, usuario, resultado, score, spoof).

**Recomendaciones**
Incluye entre 3 y 5 recomendaciones numeradas, breves y claras, basadas en los hallazgos.

Registros del bloque (hora local, usuario, resultado, score, spoof):

{logs_text}
"""

                payload = {"model": MODEL, "prompt": prompt, "stream": False}
                r = requests.post(OLLAMA_URL, json=payload, timeout=60)
                r.raise_for_status()
                response = r.json()

                # Compatibilidad con Ollama moderna
                report_block = response.get("response", "")
                if not report_block.strip():
                    report_block = "[⚠ Bloque sin respuesta del modelo]"

            except Exception as e:
                report_block = f"[❌ Error en el bloque {i+1}: {e}]"

            # Limpiar paréntesis residuales y formatear usuarios
            report_block = re.sub(r'\(([^)]+)\)', r'\1', report_block)
            report_block = re.sub(r'\busuario\s+(\d+)\b', r'usuario "\1"', report_block, flags=re.IGNORECASE)
            report_block = re.sub(r'\busuario\s+desconocido\b', r'usuario "desconocido"', report_block, flags=re.IGNORECASE)

            final_report += f"--- Bloque {i+1}/{total_blocks} ---\n{report_block}\n\n"

        return final_report.strip()

    except Exception as e:
        return f"[❌ Error general al generar el reporte: {e}]"
