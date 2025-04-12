# Actividad 4: Aprendizaje No Supervisado - Análisis de Rutas de Transporte

Este proyecto implementa un sistema de recomendación de rutas utilizando clustering no supervisado para estimar costos de transporte.

## Estructura del Proyecto

```
aprendizaje_no_supervisado_act4/
├── generate_dataset.py         # Generador de datos sintéticos
├── graph_data.py               # Configuración de grafo y parámetros
├── unsupervised_model.py       # Modelo de clustering K-Means
├── route_finder_unsupervised.py # Algoritmo de búsqueda de rutas
├── transport_unsupervised_model.pkl # Modelo entrenado
├── cluster_costs.pkl           # Mapeo de costos por cluster
├── cluster_analysis.pkl        # Análisis detallado de clusters
├── cluster_evaluation.png      # Gráficos de evaluación de clusters
└── README.md                   # Documentación del proyecto
```

## Componentes Principales

### 1. Generación de Datos
`generar_dataset_sintetico`

- Crea dataset sintético con variables:
  - Nodos (a-j)
  - Distancias (1-20 unidades)
  - Condiciones operativas (hora pico, feriado, tráfico)
  - Costos calculados con variaciones aleatorias

### 2. Modelo No Supervisado
`unsupervised_model.py`

- Preprocesamiento:
  - One-Hot Encoding para variables categóricas
  - Escalado estándar para variables numéricas
- Selección óptima del número de clusters:
  - Método del codo
  - Silhouette Score
  - Calinski-Harabasz Score
- Clustering con K-Means
- Análisis detallado de clusters:
  - Características promedio
  - Distribución de variables
  - Tamaño de clusters

### 3. Buscador de Rutas
`route_finder_unsupervised.py`

- Implementación orientada a objetos
- Validación de datos de entrada
- Ajuste de costos basado en características de clusters
- Algoritmo de Dijkstra para encontrar rutas óptimas
- Visualización de información de clusters

## Requisitos
```
pip install pandas scikit-learn joblib faker matplotlib numpy
```

## Flujo de Trabajo Completo

### 1. Generar dataset
```bash
python generate_dataset.py
```

### 2. Entrenar modelo
```bash
python unsupervised_model.py
```

### 3. Buscar ruta óptima
```bash
python route_finder_unsupervised.py
```

## Salida Esperada
```
Información de Clusters:

Cluster 0:
  Tamaño: 350 registros
  Costo promedio: $12.45
  Distancia promedio: 8.2
  % Hora pico: 25.3%
  % Feriado: 8.7%

Cluster 1:
  Tamaño: 420 registros
  Costo promedio: $18.75
  Distancia promedio: 14.6
  % Hora pico: 35.2%
  % Feriado: 12.1%

Cluster 2:
  Tamaño: 230 registros
  Costo promedio: $25.30
  Distancia promedio: 18.9
  % Hora pico: 42.8%
  % Feriado: 15.4%

Ingrese el punto de partida: a
Ingrese el punto de destino: g

La mejor ruta de (a) hacia (g) es:
a -> e -> g
Costo total estimado: $12.80
```

## Archivos Modelo
- `transport_unsupervised_model.pkl`: Modelo entrenado con K-Means
- `cluster_costs.pkl`: Costos promedio por cluster
- `cluster_analysis.pkl`: Análisis detallado de características de clusters
- `cluster_evaluation.png`: Gráficos de evaluación de clusters

## Configuración Avanzada

### Modificación del Grafo
Editar `graph_data.py`:
- Añadir/eliminar conexiones en `TRANSPORT_GRAPH['connections']`
- Modificar parámetros operativos en `KNOWLEDGE_BASE`

### Ejemplo de Configuración
```python
# En graph_data.py
TRANSPORT_GRAPH['connections'].append(('g', 'h', 7))  # Nueva conexión
KNOWLEDGE_BASE.update({
    "hora_pico": True,
    "trafico": "alto"
})
```

## Validación del Modelo
1. Generar nuevo dataset:
```bash
python generate_dataset.py
```

2. Re-entrenar modelo:
```bash
python unsupervised_model.py
```

3. Verificar análisis de clusters:
```python
import joblib
cluster_analysis = joblib.load("cluster_analysis.pkl")
print("Análisis de clusters:", cluster_analysis)
```

## Ventajas del Aprendizaje No Supervisado
- No requiere etiquetas de costos para entrenamiento
- Descubre patrones naturales en los datos
- Identifica grupos de rutas con características similares
- Adaptable a cambios en las condiciones de transporte

## Contribución
1. Clonar repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Seguir convenciones de código existentes
4. Enviar Pull Requests con cambios en ramas separadas

## Licencia
Este proyecto de clase se desarrolla bajo los lineamientos académicos de la Universidad Iberoamericana. Su uso fuera del contexto educativo requiere autorización expresa.

## Contacto
Departamento de Inteligencia Artificial Iberoamericana - Unidad 4: Aprendizaje No Supervisado [2025] - Todos los derechos académicos reservados
