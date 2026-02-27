import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pysentimiento import create_analyzer
from fpdf import FPDF
from datetime import datetime

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
print("⏳ Cargando el cerebro de IA (pysentimiento)...")
analyzer = create_analyzer(task="sentiment", lang="es")
sns.set_theme(style="whitegrid")

# ==========================================
# CLASE PARA EL REPORTE PDF
# ==========================================
class ReportePDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Reporte de Análisis de Sentimientos', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

# ==========================================
# FUNCIONES DE PROCESAMIENTO Y LÓGICA
# ==========================================

def limpiar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    return texto.strip()

def clasificar_sentimiento(texto_limpio):
    if not texto_limpio: return "Neutral"
    resultado = analyzer.predict(texto_limpio)
    mapeo = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
    return mapeo.get(resultado.output, "Neutral")

def generar_grafico(df):
    plt.figure(figsize=(10, 6))
    colores = {"Positivo": "#2ecc71", "Neutral": "#95a5a6", "Negativo": "#e74c3c"}
    sns.countplot(x='sentimiento', data=df, palette=colores, hue='sentimiento', legend=False)
    plt.title('Distribución de Sentimientos de Clientes')
    plt.savefig("reporte_sentimientos.png")
    plt.close() # Cerramos para liberar memoria

def crear_pdf(df):
    print("📄 Generando archivo PDF...")
    pdf = ReportePDF()
    pdf.add_page()
    
    # Información General
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Fecha del reporte: {datetime.now().strftime('%d/%m/%Y')}", 0, 1)
    pdf.cell(0, 10, f"Total de comentarios analizados: {len(df)}", 0, 1)
    pdf.ln(10)
    
    # Resumen Estadístico
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Resumen Ejecutivo:', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    conteo = df['sentimiento'].value_counts()
    for sent, cant in conteo.items():
        porcentaje = (cant / len(df)) * 100
        pdf.cell(0, 10, f"- {sent}: {cant} ({porcentaje:.1f}%)", 0, 1)
    
    pdf.ln(10)
    
    # Insertar el Gráfico
    pdf.image('reporte_sentimientos.png', x=10, y=None, w=180)
    
    pdf.output("Reporte_Final_Sentimientos.pdf")
    print("✅ ¡PDF generado como 'Reporte_Final_Sentimientos.pdf'!")

# ==========================================
# FLUJO PRINCIPAL
# ==========================================

def ejecutar_proyecto():
    archivo_entrada = "datos_clientes.csv"
    
    try:
        # FASE 1: Carga
        df = pd.read_csv(archivo_entrada)

        # FASE 2: Análisis
        print("🧠 Analizando comentarios...")
        df['comentario_limpio'] = df['comentario'].apply(limpiar_texto)
        df['sentimiento'] = df['comentario_limpio'].apply(clasificar_sentimiento)

        # FASE 3: Gráfico
        generar_grafico(df)

        # FASE 4: Reporte PDF
        crear_pdf(df)

        print("\n" + "="*30)
        print("🎯 PROYECTO FINALIZADO")
        print("="*30)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    ejecutar_proyecto()