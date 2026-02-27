import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pysentimiento import create_analyzer
from fpdf import FPDF
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN DEL MODELO E IA
# ==========================================
print("⏳ Inicializando modelo de IA (pysentimiento)...")
try:
    # Cargamos el analizador para español
    analyzer = create_analyzer(task="sentiment", lang="es")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    exit()

# Configuración visual de los gráficos
sns.set_theme(style="whitegrid")

# ==========================================
# 2. CLASE PARA EL REPORTE PDF
# ==========================================
class ReportePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Analisis de Satisfaccion del Cliente', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

# ==========================================
# 3. FUNCIONES DE PROCESAMIENTO (Lógica)
# ==========================================

def limpiar_texto(texto):
    """Limpia el texto eliminando caracteres especiales y pasando a minusculas."""
    if not isinstance(texto, str):
        return ""
    # Convertir a minúsculas
    texto = texto.lower()
    # Quitar tildes y caracteres especiales básicos para evitar errores de codificación
    texto = re.sub(r'[^a-z\s]', '', texto)
    return texto.strip()

def clasificar_sentimiento(texto_limpio):
    """Predice si el texto es Positivo, Negativo o Neutral."""
    if not texto_limpio or len(texto_limpio) < 2:
        return "Neutral"
    try:
        resultado = analyzer.predict(texto_limpio)
        mapeo = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
        return mapeo.get(resultado.output, "Neutral")
    except:
        return "Neutral"

# ==========================================
# 4. EJECUCIÓN PRINCIPAL DEL PROYECTO
# ==========================================

def ejecutar_proyecto():
    archivo_entrada = "datos_clientes.csv"
    
    # Verificación de que el archivo existe
    if not os.path.exists(archivo_entrada):
        print(f"⚠️ Error: No existe '{archivo_entrada}'. Ejecuta 'generar_datos.py' primero.")
        return

    try:
        # FASE 1: Carga de datos
        df = pd.read_csv(archivo_entrada)
        print(f"📊 Procesando {len(df)} comentarios...")

        # FASE 2: Análisis (Aplicamos limpieza y clasificación)
        df['texto_limpio'] = df['comentario'].apply(limpiar_texto)
        df['sentimiento'] = df['texto_limpio'].apply(clasificar_sentimiento)

        # FASE 3: Generación de Gráfico
        print("📈 Generando gráfico de barras...")
        plt.figure(figsize=(10, 6))
        colores = {"Positivo": "#2ecc71", "Neutral": "#95a5a6", "Negativo": "#e74c3c"}
        
        sns.countplot(x='sentimiento', data=df, palette=colores, hue='sentimiento', legend=False)
        plt.title('Distribucion de Sentimientos')
        plt.xlabel('Sentimiento')
        plt.ylabel('Cantidad')
        
        grafico_path = "reporte_grafico.png"
        plt.savefig(grafico_path)
        plt.close()

        # FASE 4: Generación de Reporte PDF
        print("📄 Creando reporte PDF final...")
        pdf = ReportePDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        
        # Texto del reporte
        pdf.cell(0, 10, f"Fecha del analisis: {datetime.now().strftime('%d/%m/%Y')}", 0, 1)
        pdf.cell(0, 10, f"Total de clientes analizados: {len(df)}", 0, 1)
        pdf.ln(5)
        
        # Resumen de conteos
        conteo = df['sentimiento'].value_counts()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Resumen Estadistico:", 0, 1)
        pdf.set_font('Arial', '', 12)
        
        for sent, cant in conteo.items():
            linea = f"- {sent}: {cant} comentarios"
            # Usamos encode/decode para asegurar que caracteres especiales no rompan el PDF
            pdf.cell(0, 10, linea.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
        
        pdf.ln(10)
        # Insertar imagen del gráfico
        pdf.image(grafico_path, x=15, y=None, w=180)
        
        # Guardar archivo
        nombre_pdf = "Reporte_Final_Sentimientos.pdf"
        pdf.output(nombre_pdf)
        
        print("\n" + "="*40)
        print(f"✨ ¡PROYECTO TERMINADO CON ÉXITO!")
        print(f"📁 Reporte generado: {nombre_pdf}")
        print("="*40)

    except Exception as e:
        print(f"❌ Ocurrió un error inesperado durante la ejecución: {e}")

if __name__ == "__main__":
    ejecutar_proyecto()