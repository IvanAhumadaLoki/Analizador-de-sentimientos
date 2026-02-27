import pandas as pd
import re
from pysentimiento import create_analyzer

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
# Cargamos el analizador de español (esto puede tardar unos segundos la primera vez)
print("⏳ Cargando el cerebro de IA (pysentimiento)...")
analyzer = create_analyzer(task="sentiment", lang="es")

# ==========================================
# FUNCIONES DE PROCESAMIENTO
# ==========================================

def limpiar_texto(texto):
    """
    Limpia el texto: quita caracteres especiales, números y lo pasa a minúsculas.
    """
    if not isinstance(texto, str):
        return ""
    
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    # Eliminar caracteres especiales y números (dejamos solo letras y espacios)
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    # Eliminar espacios extra
    texto = texto.strip()
    return texto

def clasificar_sentimiento(texto_limpio):
    """
    Recibe el texto y devuelve la etiqueta: Positivo, Negativo o Neutral.
    """
    if not texto_limpio:
        return "NEUTRAL"
    
    resultado = analyzer.predict(texto_limpio)
    
    # Mapeamos las etiquetas del modelo a palabras más legibles
    mapeo = {
        "POS": "Positivo",
        "NEG": "Negativo",
        "NEU": "Neutral"
    }
    return mapeo.get(resultado.output, "Neutral")

# ==========================================
# FLUJO PRINCIPAL DEL PROYECTO
# ==========================================

def ejecutar_analizador():
    archivo_entrada = "datos_clientes.csv"
    archivo_salida = "datos_analizados.csv"

    try:
        # FASE 1: Carga de datos
        print(f"📂 Leyendo el archivo: {archivo_entrada}...")
        df = pd.read_csv(archivo_entrada)

        # FASE 2: Procesamiento y Análisis
        print("🧹 Limpiando comentarios...")
        df['comentario_limpio'] = df['comentario'].apply(limpiar_texto)

        print("🧠 Analizando sentimientos (esto puede tardar según el volumen)...")
        df['sentimiento'] = df['comentario_limpio'].apply(clasificar_sentimiento)

        # Guardar resultados
        df.to_csv(archivo_salida, index=False, encoding='utf-8')
        
        # MOSTRAR RESULTADOS EN CONSOLA
        print("\n" + "="*30)
        print("✅ PROCESO COMPLETADO CON ÉXITO")
        print("="*30)
        print(f"Archivo guardado como: {archivo_salida}")
        print("\n--- RESUMEN DE RESULTADOS ---")
        print(df['sentimiento'].value_counts())
        print("="*30)

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{archivo_entrada}'. Asegúrate de correr primero 'generar_datos.py'.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    ejecutar_analizador()