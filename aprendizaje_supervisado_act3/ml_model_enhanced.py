import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# Crear directorio para guardar gráficos si no existe
os.makedirs('model_analysis', exist_ok=True)

# 1. Cargar dataset extendido
df = pd.read_csv("dataset_transporte_supervisado.csv")

# 2. Análisis exploratorio básico
print("\nAnálisis Exploratorio de Datos:")
print(f"Total de registros: {len(df)}")
print("\nEstadísticas descriptivas:")
print(df.describe())

# Visualizar distribución de costos
plt.figure(figsize=(10, 6))
sns.histplot(df['costo'], kde=True)
plt.title('Distribución de Costos de Transporte')
plt.xlabel('Costo')
plt.ylabel('Frecuencia')
plt.savefig('model_analysis/costo_distribution.png')
plt.close()

# Visualizar correlaciones (solo para columnas numéricas)
numeric_cols = ['distancia', 'hora_pico', 'feriado', 'costo']
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación (Variables Numéricas)')
plt.tight_layout()
plt.savefig('model_analysis/correlation_matrix.png')
plt.close()

# 3. Preprocesamiento con ColumnTransformer
categorical_cols = ['origen', 'destino', 'tipo_vehiculo', 'trafico']
numeric_cols = ['distancia', 'hora_pico', 'feriado']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# 4. Crear pipeline de modelo
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 5. Dividir datos
X = df.drop('costo', axis=1)
y = df['costo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Optimización de hiperparámetros
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', 
    n_jobs=-1, verbose=1
)

print("\nIniciando búsqueda de hiperparámetros...")
grid_search.fit(X_train, y_train)

print(f"\nMejores parámetros: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# 7. Evaluación con validación cruzada
cv_scores = cross_val_score(
    best_model, X, y, cv=5, scoring='neg_mean_squared_error'
)
print(f"\nMSE promedio (CV): {-cv_scores.mean():.2f} (±{cv_scores.std() * 2:.2f})")

# 8. Evaluación en conjunto de prueba
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMétricas de evaluación en conjunto de prueba:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# 9. Visualización de predicciones vs valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.tight_layout()
plt.savefig('model_analysis/predictions_vs_real.png')
plt.close()

# 10. Análisis de importancia de características
# Extraer el regresor del pipeline
regressor = best_model.named_steps['regressor']
feature_names = (
    best_model.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_cols).tolist() + 
    numeric_cols
)

# Obtener importancia de características
importances = regressor.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualizar importancia de características
plt.figure(figsize=(12, 8))
plt.title('Importancia de Características')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('model_analysis/feature_importance.png')
plt.close()

# 11. Guardar modelo y análisis
joblib.dump(best_model, 'transport_model.pkl')
joblib.dump(grid_search.best_params_, 'model_analysis/best_params.pkl')

# Guardar métricas de evaluación
metrics = {
    'mse': mse,
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'cv_mse_mean': -cv_scores.mean(),
    'cv_mse_std': cv_scores.std()
}
joblib.dump(metrics, 'model_analysis/model_metrics.pkl')

print("\nModelo y análisis guardados exitosamente.")
print("Gráficos de análisis guardados en el directorio 'model_analysis/'")