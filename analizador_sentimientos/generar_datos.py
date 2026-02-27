import pandas as pd
from faker import Faker
import random

# Inicializamos Faker en español
fake = Faker(['es_ES'])

def crear_comentarios_falsos(cantidad=50):
    datos = []
    
    # Listas de apoyo para darle "sabor" a los datos
    positivos = ["Increíble", "Muy contento", "Excelente servicio", "Lo recomiendo", "Genial"]
    negativos = ["Pésimo", "Una decepción", "Llegó tarde", "No funciona", "Mal soporte"]
    
    for _ in range(cantidad):
        tipo = random.choice(['pos', 'neg', 'neu'])
        
        if tipo == 'pos':
            texto = f"{random.choice(positivos)}. {fake.sentence()}"
        elif tipo == 'neg':
            texto = f"{random.choice(negativos)}. {fake.sentence()}"
        else:
            texto = fake.sentence()
            
        datos.append({
            "id": _ + 1,
            "cliente": fake.name(),
            "comentario": texto,
            "fecha": fake.date_this_year()
        })
    
    # Crear el DataFrame y guardarlo
    df = pd.DataFrame(datos)
    df.to_csv("datos_clientes.csv", index=False, encoding='utf-8')
    print("✅ ¡Archivo 'datos_clientes.csv' creado con éxito!")

if __name__ == "__main__":
    crear_comentarios_falsos()