import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pysentimiento import create_analyzer

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
print("⏳ Cargando el cerebro de IA (pysentimiento)...")
analyzer = create_analyzer(task="sentiment", lang="es")

# Configuración estética de los gráficos
sns.set_theme(style="whitegrid")

# ==========================================
# FUNCIONES DE PROCESAMIENTO
# ==========================================

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    return texto.strip()

def clasificar_sentimiento(texto_limpio):
    if not texto_limpio:
        return "Neutral"
    resultado = analyzer.predict(texto_limpio)
    mapeo = {"POS": "Positivo", "NEG": "Negativo", "NEU": "Neutral"}
    return mapeo.get(resultado.output, "Neutral")

# ==========================================
# FASE 3: VISUALIZACIÓN
# ==========================================

def generar_grafico(df):
    """
    Crea un gráfico de barras con los resultados y lo guarda como imagen.
    """
    print("📊 Generando gráfico de resultados...")
    
    plt.figure(figsize=(10, 6))
    
    # Definir colores fijos para coherencia (Verde, Gris, Rojo)
    colores = {"Positivo": "#2ecc71", "Neutral": "#95a5a6", "Negativo": "#e74c3c"}
    
    # Crear el gráfico de barras
    sns.countplot(x='sentimiento', data=df, palette=colores, hue='sentimiento', legend=False)
    
    plt.title('Distribución de Sentimientos de Clientes', fontsize=16)
    plt.xlabel('Sentimiento', fontsize=12)
    plt.ylabel('Cantidad de Comentarios', fontsize=12)
    
    # Guardar el gráfico como imagen para el reporte futuro
    plt.savefig("reporte_sentimientos.png")
    print("✅ Gráfico guardado como 'reporte_sentimientos.png'")
    
    # Mostrar el gráfico en una ventana (opcional)
    # plt.show()

# ==========================================
# FLUJO PRINCIPAL
# ==========================================

def ejecutar_analizador():
    archivo_entrada = "datos_clientes.csv"
    
    try:
        # FASE 1: Carga
        df = pd.read_csv(archivo_entrada)

        # FASE 2: Análisis
        print("🧹 Procesando y analizando comentarios...")
        df['comentario_limpio'] = df['comentario'].apply(limpiar_texto)
        df['sentimiento'] = df['comentario_limpio'].apply(clasificar_sentimiento)

        # FASE 3: Visualización
        generar_grafico(df)

        # Guardar CSV final
        df.to_csv("datos_analizados.csv", index=False, encoding='utf-8')
        
        print("\n" + "="*30)
        print("✅ FASE 3 COMPLETADA")
        print("="*30)
        print(df['sentimiento'].value_counts())

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    ejecutar_analizador()