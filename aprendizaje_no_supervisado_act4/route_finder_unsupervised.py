# route_finder_unsupervised.py
import heapq
import joblib
from graph_data import TRANSPORT_GRAPH, KNOWLEDGE_BASE
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import json
import os
from datetime import datetime

# Configuración de logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/route_finder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RouteFinderUnsupervised:
    def __init__(self, model_path='transport_unsupervised_model.pkl'):
        self.model = None
        self.scaler = None
        self.cluster_stats = None
        self.model_path = model_path
        self.route_history = []
        self.load_model()
        self.load_history()
        
    def load_model(self):
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.cluster_stats = model_data['cluster_stats']
            logger.info("Modelo no supervisado cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            self.model = None
            self.scaler = None
            self.cluster_stats = None
            
    def load_history(self):
        """Carga el historial de rutas desde un archivo JSON si existe."""
        try:
            if os.path.exists('route_history.json'):
                with open('route_history.json', 'r') as f:
                    self.route_history = json.load(f)
                logger.info(f"Historial de rutas cargado: {len(self.route_history)} entradas")
        except Exception as e:
            logger.warning(f"No se pudo cargar el historial de rutas: {str(e)}")
            self.route_history = []
            
    def save_history(self):
        """Guarda el historial de rutas en un archivo JSON."""
        try:
            with open('route_history.json', 'w') as f:
                json.dump(self.route_history, f, indent=2, default=str)
            logger.info("Historial de rutas guardado")
        except Exception as e:
            logger.error(f"Error al guardar el historial de rutas: {str(e)}")
            
    def predict_cluster(self, features):
        if self.model is None or self.scaler is None:
            logger.error("El modelo no está cargado")
            return None
            
        try:
            # Escalar las características
            scaled_features = self.scaler.transform([features])
            # Predecir el cluster
            cluster = self.model.predict(scaled_features)[0]
            return cluster
        except Exception as e:
            logger.error(f"Error al predecir el cluster: {str(e)}")
            return None
            
    def get_route_recommendations(self, origin, destination, conditions=None):
        if conditions is None:
            conditions = KNOWLEDGE_BASE.copy()
            
        # Crear características para la predicción
        features = [
            conditions.get('distancia', 0),
            1 if conditions.get('hora_pico', False) else 0,
            1 if conditions.get('feriado', False) else 0
        ]
        
        # Predecir el cluster
        cluster = self.predict_cluster(features)
        if cluster is None:
            return None
            
        # Obtener recomendaciones basadas en el cluster
        recommendations = self._get_cluster_recommendations(cluster, origin, destination)
        
        # Guardar la ruta en el historial
        if recommendations:
            self._save_route_to_history(origin, destination, conditions, recommendations)
            
        return recommendations
        
    def _get_cluster_recommendations(self, cluster, origin, destination):
        try:
            # Obtener estadísticas del cluster
            cluster_data = self.cluster_stats.loc[cluster]
            
            # Encontrar la ruta
            path = self._find_optimal_path(origin, destination, cluster_data)
            
            # Calcular costo estimado basado en estadísticas del cluster
            base_cost = cluster_data[('distancia', 'mean')] * 10
            traffic_factor = 1.2 if cluster_data[('hora_pico', 'mean')] > 0.5 else 1.0
            holiday_factor = 1.3 if cluster_data[('feriado', 'mean')] > 0.5 else 1.0
            
            estimated_cost = base_cost * traffic_factor * holiday_factor
            
            return {
                'cluster': cluster,
                'path': path,
                'estimated_cost': round(estimated_cost, 2),
                'traffic_factor': round(traffic_factor, 2),
                'holiday_factor': round(holiday_factor, 2),
                'cluster_stats': {
                    'avg_distance': round(cluster_data[('distancia', 'mean')], 2),
                    'peak_hour_prob': round(cluster_data[('hora_pico', 'mean')], 2),
                    'holiday_prob': round(cluster_data[('feriado', 'mean')], 2)
                }
            }
        except Exception as e:
            logger.error(f"Error al obtener recomendaciones: {str(e)}")
            return None
            
    def _find_optimal_path(self, origin, destination, cluster_data):
        # Implementación del algoritmo A* para encontrar la ruta óptima
        graph = {node: [] for node in TRANSPORT_GRAPH['nodes']}
        weights = {}
        
        # Construir grafo con pesos
        for start, end, weight in TRANSPORT_GRAPH['connections']:
            graph[start].append(end)
            graph[end].append(start)
            weights[(start, end)] = weight
            weights[(end, start)] = weight
            
        # Ajustar pesos basados en estadísticas del cluster
        for (start, end), weight in weights.items():
            if cluster_data[('hora_pico', 'mean')] > 0.5:
                weights[(start, end)] *= 1.2
            if cluster_data[('feriado', 'mean')] > 0.5:
                weights[(start, end)] *= 1.3
                
        # Implementación de A*
        open_set = {origin}
        closed_set = set()
        came_from = {}
        g_score = {node: float('inf') for node in graph}
        g_score[origin] = 0
        f_score = {node: float('inf') for node in graph}
        f_score[origin] = self._heuristic(origin, destination)
        
        while open_set:
            current = min(open_set, key=lambda x: f_score[x])
            
            if current == destination:
                return self._reconstruct_path(came_from, current)
                
            open_set.remove(current)
            closed_set.add(current)
            
            for neighbor in graph[current]:
                if neighbor in closed_set:
                    continue
                    
                tentative_g_score = g_score[current] + weights[(current, neighbor)]
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue
                    
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, destination)
                
        return None
        
    def _heuristic(self, node, goal):
        # Heurística simple basada en la distancia entre nodos
        return 1  # Podría mejorarse con una heurística más sofisticada
        
    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
        
    def _save_route_to_history(self, origin, destination, conditions, recommendations):
        route_entry = {
            'timestamp': datetime.now().isoformat(),
            'origin': origin,
            'destination': destination,
            'conditions': conditions,
            'cost': recommendations['estimated_cost'],
            'path': recommendations['path'],
            'cluster': recommendations['cluster']
        }
        self.route_history.append(route_entry)
        
        # Guardar historial en archivo
        try:
            with open('route_history.json', 'w') as f:
                json.dump(self.route_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error al guardar historial de rutas: {str(e)}")
            
    def get_route_history(self):
        return self.route_history
        
    def export_history_to_csv(self, filename='route_history.csv'):
        """Exporta el historial de rutas a un archivo CSV."""
        if not self.route_history:
            logger.warning("No hay historial para exportar")
            return
            
        try:
            df = pd.DataFrame(self.route_history)
            df.to_csv(filename, index=False)
            logger.info(f"Historial exportado a {filename}")
        except Exception as e:
            logger.error(f"Error al exportar historial: {str(e)}")
            
    def get_route_statistics(self):
        """Calcula estadísticas sobre el historial de rutas."""
        if not self.route_history:
            return {
                'total_routes': 0,
                'average_cost': 0,
                'min_cost': 0,
                'max_cost': 0,
                'most_common_origin': 'N/A',
                'most_common_destination': 'N/A'
            }
            
        costs = [route['cost'] for route in self.route_history]
        origins = [route['origin'] for route in self.route_history]
        destinations = [route['destination'] for route in self.route_history]
        
        return {
            'total_routes': len(self.route_history),
            'average_cost': sum(costs) / len(costs) if costs else 0,
            'min_cost': min(costs) if costs else 0,
            'max_cost': max(costs) if costs else 0,
            'most_common_origin': max(set(origins), key=origins.count) if origins else 'N/A',
            'most_common_destination': max(set(destinations), key=destinations.count) if destinations else 'N/A'
        }

def main():
    try:
        finder = RouteFinderUnsupervised()
        
        # Mostrar estadísticas si hay historial
        if finder.get_route_history():
            stats = finder.get_route_statistics()
            print("\nEstadísticas de búsquedas anteriores:")
            print(f"  Total de rutas buscadas: {stats['total_routes']}")
            print(f"  Costo promedio: ${stats['average_cost']:.2f}")
            print(f"  Ruta más económica: ${stats['min_cost']:.2f}")
            print(f"  Ruta más costosa: ${stats['max_cost']:.2f}")
            print(f"  Origen más común: {stats['most_common_origin']}")
            print(f"  Destino más común: {stats['most_common_destination']}")
        
        while True:
            print("\n=== SISTEMA DE BÚSQUEDA DE RUTAS ÓPTIMAS (NO SUPERVISADO) ===")
            print("1. Buscar nueva ruta")
            print("2. Ver historial de rutas")
            print("3. Exportar historial a CSV")
            print("4. Cambiar condiciones de búsqueda")
            print("5. Salir")
            
            opcion = input("\nSeleccione una opción (1-5): ")
            
            if opcion == "1":
                origin = input("Ingrese el origen (a-j): ").lower()
                destination = input("Ingrese el destino (a-j): ").lower()
                
                if origin not in TRANSPORT_GRAPH['nodes'] or destination not in TRANSPORT_GRAPH['nodes']:
                    print("Nodos inválidos. Use letras de a a j.")
                    continue
                    
                recommendations = finder.get_route_recommendations(origin, destination)
                
                if recommendations:
                    print("\nRecomendaciones:")
                    print(f"Cluster: {recommendations['cluster']}")
                    print(f"Ruta: {' -> '.join(recommendations['path'])}")
                    print(f"Costo estimado: ${recommendations['estimated_cost']}")
                    print(f"Factor de tráfico: {recommendations['traffic_factor']}")
                    print(f"Factor de feriado: {recommendations['holiday_factor']}")
                    print("\nEstadísticas del cluster:")
                    for key, value in recommendations['cluster_stats'].items():
                        print(f"  {key}: {value}")
                else:
                    print("No se pudieron generar recomendaciones.")
                    
            elif opcion == "2":
                history = finder.get_route_history()
                if not history:
                    print("\nNo hay historial de rutas.")
                    continue
                
                print("\nHistorial de rutas:")
                for i, route in enumerate(history[-10:], 1):  # Mostrar las últimas 10 rutas
                    print(f"{i}. {route['origin']} -> {route['destination']} (${route['cost']:.2f})")
                    
            elif opcion == "3":
                filename = input("Ingrese el nombre del archivo (default: route_history.csv): ") or "route_history.csv"
                finder.export_history_to_csv(filename)
                print(f"Historial exportado a {filename}")
                
            elif opcion == "4":
                print("\nCondiciones actuales:")
                print(f"  Hora pico: {KNOWLEDGE_BASE.get('hora_pico', False)}")
                print(f"  Feriado: {KNOWLEDGE_BASE.get('feriado', False)}")
                print(f"  Tipo de vehículo: {KNOWLEDGE_BASE.get('tipo_vehiculo', 'autobus')}")
                print(f"  Tráfico: {KNOWLEDGE_BASE.get('trafico', 'medio')}")
                
                print("\n¿Desea cambiar alguna condición? (s/n): ")
                if input().lower() == 's':
                    KNOWLEDGE_BASE['hora_pico'] = input("¿Es hora pico? (s/n): ").lower() == 's'
                    KNOWLEDGE_BASE['feriado'] = input("¿Es feriado? (s/n): ").lower() == 's'
                    KNOWLEDGE_BASE['tipo_vehiculo'] = input("Tipo de vehículo (autobus/taxi/carga/particular): ").lower()
                    KNOWLEDGE_BASE['trafico'] = input("Nivel de tráfico (bajo/medio/alto): ").lower()
                    print("Condiciones actualizadas.")
                    
            elif opcion == "5":
                print("¡Hasta luego!")
                break
                
            else:
                print("Opción no válida. Por favor, intente de nuevo.")
                
    except Exception as e:
        logger.error(f"Error en la ejecución principal: {str(e)}")
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
