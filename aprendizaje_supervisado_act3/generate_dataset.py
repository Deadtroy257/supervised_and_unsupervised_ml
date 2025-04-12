import pandas as pd
import numpy as np
from faker import Faker

def generar_dataset_sintetico(num_registros=1000):
    fake = Faker()
    np.random.seed(42)
    
    # Configuración base
    nodos = [chr(97+i) for i in range(10)]  # Nodos de la 'a' a la 'j'
    tipos_vehiculo = ['autobus', 'taxi', 'carga', 'particular']
    niveles_trafico = ['bajo', 'medio', 'alto']
    
    dataset = []
    for _ in range(num_registros):
        origen, destino = np.random.choice(nodos, size=2, replace=False)
        distancia = np.random.randint(1, 20)
        hora_pico = np.random.choice([0, 1], p=[0.7, 0.3])
        feriado = np.random.choice([0, 1], p=[0.9, 0.1])
        tipo_vehiculo = np.random.choice(tipos_vehiculo)
        trafico = np.random.choice(niveles_trafico, p=[0.6, 0.3, 0.1])
        
        # Cálculo base del costo con variaciones
        costo_base = distancia * (1 + 0.2*hora_pico + 0.1*feriado)
        
        # Ajustes por tipo de vehículo
        if tipo_vehiculo == 'taxi': costo_base *= 1.3
        elif tipo_vehiculo == 'carga': costo_base *= 1.5
        elif tipo_vehiculo == 'particular': costo_base *= 0.9
        
        # Ajustes por tráfico
        if trafico == 'medio': costo_base *= 1.1
        elif trafico == 'alto': costo_base *= 1.3
        
        # Ruido aleatorio para variación
        costo_final = costo_base * np.random.uniform(0.95, 1.05)
        
        dataset.append([
            origen,
            destino,
            distancia,
            hora_pico,
            feriado,
            tipo_vehiculo,
            trafico,
            round(costo_final, 1)
        ])
    
    return pd.DataFrame(
        dataset,
        columns=['origen', 'destino', 'distancia', 'hora_pico', 'feriado', 
                 'tipo_vehiculo', 'trafico', 'costo']
    )

# Generar y guardar dataset
df = generar_dataset_sintetico(1000)
df.to_csv('dataset_transporte_supervisado.csv', index=False)
print(f"Dataset generado con {len(df)} registros")