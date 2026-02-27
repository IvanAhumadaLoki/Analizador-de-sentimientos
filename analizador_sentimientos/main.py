import pandas as pd

def cargar_datos(ruta):
    try:
        df = pd.read_csv(ruta)
        print("--- Datos cargados correctamente ---")
        print(df.head()) # Muestra las primeras 5 filas
        return df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")

if __name__ == "__main__":
    datos = cargar_datos("datos_clientes.csv")