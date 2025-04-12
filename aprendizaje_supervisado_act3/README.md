# Actividad 3: Aprendizaje Supervisado - Optimización de Rutas de Transporte

Sistema de recomendación de rutas utilizando modelos de machine learning supervisado para predecir costos de transporte.

## Estructura del Proyecto
```
aprendizaje_supervisado_act3/
├── generate_dataset.py        # Generador de datos sintéticos
├── graph_data.py              # Configuración del grafo y parámetros
├── ml_model_enhanced.py       # Entrenamiento del modelo predictivo
├── route_finder.py            # Sistema de búsqueda de rutas óptimas
├── transport_model.pkl        # Modelo de Random Forest entrenado
├── model_analysis/            # Directorio con análisis y visualizaciones
├── logs/                      # Directorio con logs del sistema
├── route_history.json         # Historial de búsquedas de rutas
└── README.md                  # Documentación del sistema
```

## Componentes Principales

### 1. Generación de Datos Sintéticos
`generar_dataset_sintetico`

Variables generadas:
* 10 nodos (a-j)
* Distancias entre 1-20 unidades
* Condiciones de tráfico (bajo/medio/alto)
* Tipos de vehículo (autobus/taxi/carga/particular)
* Cálculo de costos con variaciones aleatorias

### 2. Modelo Predictivo Mejorado
`modelo_predictivo_enhanced`

Pipeline de procesamiento:
* One-Hot Encoding para características categóricas
* StandardScaler para características numéricas
* Random Forest Regressor con optimización de hiperparámetros
* Validación cruzada para evaluación robusta
* Análisis de importancia de características
* Visualizaciones de resultados y métricas

Métricas de evaluación:
* MSE (Error Cuadrático Medio)
* RMSE (Raíz del Error Cuadrático Medio)
* MAE (Error Absoluto Medio)
* R² (Coeficiente de determinación)

### 3. Sistema de Rutas Inteligente
`search_best_route_ml`

Flujo de trabajo:
1. Predicción de costos usando transport_model.pkl
2. Construcción de grafo con costos predichos
3. Algoritmo de Dijkstra para encontrar ruta óptima
4. Registro de búsquedas en historial
5. Estadísticas de uso del sistema

Características adicionales:
* Sistema de logging para seguimiento de operaciones
* Historial de búsquedas con exportación a CSV
* Interfaz de usuario mejorada con menú interactivo
* Capacidad para cambiar condiciones de búsqueda
* Validación mejorada de entradas

## Requisitos de Instalación
```
pip install pandas scikit-learn joblib faker matplotlib seaborn numpy
```

## Flujo de Trabajo Completo
```bash
# 1. Generar dataset sintético
python generate_dataset.py

# 2. Entrenar modelo predictivo
python ml_model_enhanced.py

# 3. Buscar ruta óptima
python route_finder.py
```

## Ejemplo de Uso

### Modificar parámetros en graph_data.py
```python
KNOWLEDGE_BASE.update({
    "hora_pico": True,
    "trafico": "alto"
})
```

### Ejecutar sistema
```
>>> python route_finder.py
=== SISTEMA DE BÚSQUEDA DE RUTAS ÓPTIMAS ===
1. Buscar nueva ruta
2. Ver historial de rutas
3. Exportar historial a CSV
4. Cambiar condiciones de búsqueda
5. Salir

Seleccione una opción (1-5): 1
Ingrese el punto de partida: a
Ingrese el punto de destino: g

La mejor ruta de (a) hacia (g) es:
a -> e -> g
Costo total estimado: $15.2
```

## Configuración Avanzada

### Modificar conexiones en graph_data.py
```python
TRANSPORT_GRAPH['connections'].append(('g', 'h', 7))  # Nueva conexión
```

### Validación del Modelo
```python
import joblib
model = joblib.load("transport_model.pkl")
print(model.n_features_in_)  # Debe mostrar 7 características
```

## Mejoras Implementadas

### 1. Modelo de Machine Learning
- Optimización de hiperparámetros con GridSearchCV
- Validación cruzada para evaluación robusta
- Análisis de importancia de características
- Visualizaciones de resultados y métricas
- Guardado de métricas y parámetros óptimos

### 2. Sistema de Rutas
- Sistema de logging para seguimiento de operaciones
- Historial de búsquedas con exportación a CSV
- Interfaz de usuario mejorada con menú interactivo
- Capacidad para cambiar condiciones de búsqueda
- Validación mejorada de entradas
- Estadísticas de uso del sistema

### 3. Documentación
- README actualizado con nuevas funcionalidades
- Documentación de código mejorada
- Ejemplos de uso actualizados

## Licencia
Proyecto académico bajo licencia educativa de la Universidad Iberoamericana. Prohibido su uso comercial sin autorización.

## Contacto
Departamento de Inteligencia Artificial Iberoamericana- Unidad 3: Aprendizaje Supervisado [2025] - Todos los derechos académicos reservados