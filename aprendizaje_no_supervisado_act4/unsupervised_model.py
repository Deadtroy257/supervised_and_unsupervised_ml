# unsupervised_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import seaborn as sns
import os

# Crear directorios para guardar resultados
os.makedirs('model_analysis', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# 1. Cargar el dataset sin la columna 'costo' para el clustering
df = pd.read_csv("dataset_transporte_no_supervisado.csv")

# Separamos las columnas categóricas y numéricas que usaremos para el clustering
categorical_cols = ['origen', 'destino', 'tipo_vehiculo', 'trafico']
numeric_cols = ['distancia', 'hora_pico', 'feriado']

# 2. Crear pipeline de preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ]
)

def load_and_preprocess_data():
    # Cargar datos
    df = pd.read_csv('dataset_transporte_no_supervisado.csv')
    
    # Seleccionar características para clustering
    features = ['distancia', 'hora_pico', 'feriado']
    
    # Convertir variables categóricas a numéricas
    df['hora_pico'] = df['hora_pico'].astype(int)
    df['feriado'] = df['feriado'].astype(int)
    
    # Escalar características
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    return df, X, scaler

def find_optimal_clusters(X, max_clusters=10):
    # Métricas para diferentes números de clusters
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
        davies_scores.append(davies_bouldin_score(X, kmeans.labels_))
    
    # Visualizar métricas
    plt.figure(figsize=(15, 10))
    
    # Inertia
    plt.subplot(2, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, 'bo-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inertia')
    plt.title('Método del Codo')
    
    # Silhouette Score
    plt.subplot(2, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    
    # Calinski-Harabasz Score
    plt.subplot(2, 2, 3)
    plt.plot(range(2, max_clusters + 1), calinski_scores, 'go-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score')
    
    # Davies-Bouldin Score
    plt.subplot(2, 2, 4)
    plt.plot(range(2, max_clusters + 1), davies_scores, 'mo-')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score')
    
    plt.tight_layout()
    plt.savefig('visualizations/cluster_metrics.png')
    plt.close()
    
    # Normalizar scores para comparación
    normalized_scores = {
        'silhouette': silhouette_scores / max(silhouette_scores),
        'calinski': calinski_scores / max(calinski_scores),
        'davies': [1 - (score / max(davies_scores)) for score in davies_scores]
    }
    
    # Combinar scores para encontrar óptimo
    combined_scores = [
        (s + c + d) / 3 
        for s, c, d in zip(
            normalized_scores['silhouette'],
            normalized_scores['calinski'],
            normalized_scores['davies']
        )
    ]
    
    optimal_k = combined_scores.index(max(combined_scores)) + 2
    return optimal_k

def train_model(X, n_clusters):
    # Entrenar modelo final
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model

def analyze_clusters(df, model, scaler):
    # Añadir etiquetas de cluster al DataFrame
    df['cluster'] = model.labels_
    
    # Análisis por cluster
    cluster_stats = df.groupby('cluster').agg({
        'distancia': ['mean', 'std', 'min', 'max'],
        'hora_pico': 'mean',
        'feriado': 'mean'
    }).round(2)
    
    # Guardar estadísticas
    cluster_stats.to_csv('model_analysis/cluster_statistics.csv')
    
    # Visualizar distribución de clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cluster')
    plt.title('Distribución de Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Frecuencia')
    plt.savefig('visualizations/cluster_distribution.png')
    plt.close()
    
    # Visualizar características por cluster
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df, x='cluster', y='distancia')
    plt.title('Distancia por Cluster')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='cluster', y='hora_pico')
    plt.title('Hora Pico por Cluster')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data=df, x='cluster', y='feriado')
    plt.title('Feriado por Cluster')
    
    plt.tight_layout()
    plt.savefig('visualizations/cluster_characteristics.png')
    plt.close()
    
    return cluster_stats

def main():
    # Cargar y preprocesar datos
    print("Cargando y preprocesando datos...")
    df, X, scaler = load_and_preprocess_data()
    
    # Encontrar número óptimo de clusters
    print("Buscando número óptimo de clusters...")
    optimal_k = find_optimal_clusters(X)
    print(f"Número óptimo de clusters: {optimal_k}")
    
    # Entrenar modelo final
    print("Entrenando modelo final...")
    model = train_model(X, optimal_k)
    
    # Analizar clusters
    print("Analizando clusters...")
    cluster_stats = analyze_clusters(df, model, scaler)
    
    # Guardar modelo y escalador
    print("Guardando modelo y escalador...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'optimal_k': optimal_k,
        'cluster_stats': cluster_stats
    }
    joblib.dump(model_data, 'transport_unsupervised_model.pkl')
    
    print("¡Proceso completado!")
    print("\nEstadísticas de clusters:")
    print(cluster_stats)

if __name__ == "__main__":
    main()
