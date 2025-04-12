# Sistema de Análisis de Rutas de Transporte con Aprendizaje Automático

Este proyecto implementa dos enfoques diferentes para el análisis y recomendación de rutas de transporte utilizando técnicas de aprendizaje automático: un modelo supervisado y un modelo no supervisado.

## Estructura del Proyecto

```
.
├── aprendizaje_supervisado_act3/     # Solución con aprendizaje supervisado
│   ├── dataset_transporte_supervisado.csv
│   ├── generate_dataset.py
│   ├── graph_data.py
│   ├── ml_model_enhanced.py
│   ├── route_finder.py
│   ├── transport_model.pkl
│   └── README.md
│
├── aprendizaje_no_supervisado_act4/  # Solución con aprendizaje no supervisado
│   ├── dataset_transporte_no_supervisado.csv
│   ├── generate_dataset.py
│   ├── graph_data.py
│   ├── route_finder_unsupervised.py
│   ├── transport_unsupervised_model.pkl
│   ├── unsupervised_model.py
│   └── README.md
│
├── requirements.txt                  # Dependencias del proyecto
└── .gitignore                       # Archivos ignorados por Git
```

## Descripción General

### Aprendizaje Supervisado (Actividad 3)
- Utiliza un modelo de Random Forest para predecir el costo exacto de una ruta
- Requiere datos etiquetados (con costos) para el entrenamiento
- Proporciona predicciones precisas de costos basadas en características de la ruta
- Implementa un algoritmo A* para encontrar la ruta óptima

### Aprendizaje No Supervisado (Actividad 4)
- Utiliza KMeans para agrupar rutas similares en clusters
- No requiere datos etiquetados (sin costos) para el entrenamiento
- Descubre patrones en los datos y agrupa rutas con características similares
- Proporciona recomendaciones basadas en estadísticas de clusters

## Requisitos

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
faker>=8.0.0
```

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Modelo Supervisado

1. Generar dataset:
```bash
cd aprendizaje_supervisado_act3
python generate_dataset.py
```

2. Entrenar modelo:
```bash
python ml_model_enhanced.py
```

3. Usar buscador de rutas:
```bash
python route_finder.py
```

### Modelo No Supervisado

1. Generar dataset:
```bash
cd aprendizaje_no_supervisado_act4
python generate_dataset.py
```

2. Entrenar modelo:
```bash
python unsupervised_model.py
```

3. Usar buscador de rutas:
```bash
python route_finder_unsupervised.py
```

## Comparación de Enfoques

### Ventajas del Aprendizaje Supervisado
- Mayor precisión en la predicción de costos
- Resultados más directos y fáciles de interpretar
- Mejor para decisiones financieras específicas

### Ventajas del Aprendizaje No Supervisado
- No requiere datos etiquetados (costos históricos)
- Descubre patrones ocultos en los datos
- Más adaptable a cambios en los patrones de transporte

## Visualizaciones y Análisis

Ambas soluciones generan visualizaciones y análisis que se guardan en:
- `aprendizaje_supervisado_act3/model_analysis/`
- `aprendizaje_no_supervisado_act4/model_analysis/` y `visualizations/`

## Autores

- Diego Castro (Iberoamericana)