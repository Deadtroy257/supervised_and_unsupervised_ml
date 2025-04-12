# route_finder_ml.py

import pandas as pd
import numpy as np
import heapq
import joblib
import logging
import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Any
from graph_data import TRANSPORT_GRAPH, KNOWLEDGE_BASE

# Configurar logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/route_finder.log'),
        logging.StreamHandler()
    ]
)

class RouteFinderML:
    def __init__(self, model_path: str = "transport_model.pkl"):
        """
        Inicializa el buscador de rutas con el modelo de ML.
        
        Args:
            model_path: Ruta al archivo del modelo entrenado
        """
        try:
            self.ml_model = joblib.load(model_path)
            self.feature_names = self.ml_model.feature_names_in_
            self.route_history = []
            self.load_history()
            logging.info(f"Modelo cargado exitosamente desde {model_path}")
            logging.info(f"Características del modelo: {self.feature_names}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")
            raise RuntimeError(f"Error al cargar el modelo: {str(e)}")

    def load_history(self):
        """Carga el historial de rutas desde un archivo JSON si existe."""
        try:
            if os.path.exists('route_history.json'):
                with open('route_history.json', 'r') as f:
                    self.route_history = json.load(f)
                logging.info(f"Historial de rutas cargado: {len(self.route_history)} entradas")
        except Exception as e:
            logging.warning(f"No se pudo cargar el historial de rutas: {str(e)}")
            self.route_history = []

    def save_history(self):
        """Guarda el historial de rutas en un archivo JSON."""
        try:
            with open('route_history.json', 'w') as f:
                json.dump(self.route_history, f, indent=2, default=str)
            logging.info("Historial de rutas guardado")
        except Exception as e:
            logging.error(f"Error al guardar el historial de rutas: {str(e)}")

    def validate_input(self, origen: str, destino: str, distancia: float) -> None:
        """
        Valida los datos de entrada.
        
        Args:
            origen: Punto de origen
            destino: Punto de destino
            distancia: Distancia entre puntos
        """
        if not isinstance(distancia, (int, float)) or distancia <= 0:
            raise ValueError("La distancia debe ser un número positivo")
        if not isinstance(origen, str) or not isinstance(destino, str):
            raise ValueError("Origen y destino deben ser strings")
        if origen == destino:
            raise ValueError("Origen y destino no pueden ser iguales")
        if origen not in TRANSPORT_GRAPH["nodes"] or destino not in TRANSPORT_GRAPH["nodes"]:
            raise ValueError("Origen o destino no encontrados en el grafo de transporte")

    def predict_cost_ml(self, origen: str, destino: str, distancia: float, 
                       hora_pico: bool, feriado: bool, tipo_vehiculo: str, 
                       trafico: str) -> float:
        """
        Prepara las características y utiliza el modelo para predecir el costo.
        
        Args:
            origen: Punto de origen
            destino: Punto de destino
            distancia: Distancia entre puntos
            hora_pico: Indica si es hora pico
            feriado: Indica si es feriado
            tipo_vehiculo: Tipo de vehículo a utilizar
            trafico: Nivel de tráfico
            
        Returns:
            float: Costo predicho
        """
        self.validate_input(origen, destino, distancia)
        
        input_data = pd.DataFrame({
            'origen': [origen],
            'destino': [destino],
            'distancia': [distancia],
            'hora_pico': [hora_pico],
            'feriado': [feriado],
            'tipo_vehiculo': [tipo_vehiculo],
            'trafico': [trafico]
        })
        
        try:
            predicted_cost = self.ml_model.predict(input_data)[0]
            logging.debug(f"Predicción de costo para {origen}->{destino}: ${predicted_cost:.2f}")
            return float(predicted_cost)
        except Exception as e:
            logging.error(f"Error en la predicción para {origen}->{destino}: {str(e)}")
            raise RuntimeError(f"Error en la predicción: {str(e)}")

    def build_graph_ml(self, connections: List[Tuple], kb: Dict) -> Dict:
        """
        Construye un grafo de adyacencia utilizando el modelo de ML.
        
        Args:
            connections: Lista de tuplas (origen, destino, distancia)
            kb: Base de conocimiento con condiciones globales
            
        Returns:
            Dict: Grafo construido
        """
        graph = {}
        hora_pico = kb.get("hora_pico", False)
        feriado = kb.get("feriado", False)
        tipo_vehiculo = kb.get("tipo_vehiculo", "autobus")
        trafico = kb.get("trafico", "medio")
        
        logging.info(f"Construyendo grafo con condiciones: hora_pico={hora_pico}, feriado={feriado}, "
                    f"tipo_vehiculo={tipo_vehiculo}, trafico={trafico}")
        
        for s, e, distancia in connections:
            try:
                costo_predicho = self.predict_cost_ml(
                    s, e, distancia, hora_pico, feriado, tipo_vehiculo, trafico
                )
                
                if s not in graph:
                    graph[s] = []
                graph[s].append((e, costo_predicho))
            except Exception as e:
                logging.warning(f"Error al procesar conexión {s}->{e}: {str(e)}")
                continue
        
        logging.info(f"Grafo construido con {len(graph)} nodos")
        return graph

    @staticmethod
    def find_best_route(start: str, end: str, graph: Dict) -> Tuple[Optional[List], Optional[float]]:
        """
        Implementa el algoritmo de Dijkstra para encontrar la ruta óptima.
        
        Args:
            start: Punto de inicio
            end: Punto de destino
            graph: Grafo de conexiones
            
        Returns:
            Tuple[List, float]: Ruta óptima y su costo
        """
        heap = []
        heapq.heappush(heap, (0, start, [start]))
        visited = set()
        distances = {start: 0}
        paths = {start: [start]}
        
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node == end:
                return path, cost
            if node not in visited:
                visited.add(node)
                for neighbor, neighbor_cost in graph.get(node, []):
                    if neighbor not in visited:
                        new_cost = cost + neighbor_cost
                        new_path = path + [neighbor]
                        if neighbor not in distances or new_cost < distances[neighbor]:
                            distances[neighbor] = new_cost
                            paths[neighbor] = new_path
                            heapq.heappush(heap, (new_cost, neighbor, new_path))
        return None, None

    def search_best_route_ml(self, initial_point: str, final_point: str) -> Tuple[Optional[List], Optional[float]]:
        """
        Busca la mejor ruta entre dos puntos usando el modelo de ML.
        
        Args:
            initial_point: Punto de inicio
            final_point: Punto de destino
            
        Returns:
            Tuple[List, float]: Mejor ruta y su costo
        """
        try:
            logging.info(f"Buscando ruta de {initial_point} a {final_point}")
            
            # Construir el grafo usando el modelo de ML para predecir costos
            graph = self.build_graph_ml(
                TRANSPORT_GRAPH["connections"],
                KNOWLEDGE_BASE
            )
            
            # Buscar la mejor ruta
            route, cost = self.find_best_route(initial_point, final_point, graph)
            
            # Registrar la búsqueda en el historial
            if route:
                self.route_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'origin': initial_point,
                    'destination': final_point,
                    'route': route,
                    'cost': cost,
                    'conditions': {
                        'hora_pico': KNOWLEDGE_BASE.get("hora_pico", False),
                        'feriado': KNOWLEDGE_BASE.get("feriado", False),
                        'tipo_vehiculo': KNOWLEDGE_BASE.get("tipo_vehiculo", "autobus"),
                        'trafico': KNOWLEDGE_BASE.get("trafico", "medio")
                    }
                })
                self.save_history()
                logging.info(f"Ruta encontrada: {' -> '.join(route)} con costo ${cost:.2f}")
            else:
                logging.warning(f"No se encontró ruta de {initial_point} a {final_point}")
            
            return route, cost
        except Exception as e:
            logging.error(f"Error en la búsqueda de ruta: {str(e)}")
            return None, None

    def get_route_history(self) -> List[Dict]:
        """
        Obtiene el historial de rutas buscadas.
        
        Returns:
            List[Dict]: Historial de rutas
        """
        return self.route_history

    def get_route_statistics(self) -> Dict[str, Any]:
        """
        Calcula estadísticas sobre las rutas buscadas.
        
        Returns:
            Dict: Estadísticas de rutas
        """
        if not self.route_history:
            return {}
        
        costs = [route['cost'] for route in self.route_history if route['cost'] is not None]
        return {
            'total_routes': len(self.route_history),
            'average_cost': np.mean(costs) if costs else 0,
            'min_cost': min(costs) if costs else 0,
            'max_cost': max(costs) if costs else 0,
            'most_common_origin': pd.Series([r['origin'] for r in self.route_history]).mode().iloc[0],
            'most_common_destination': pd.Series([r['destination'] for r in self.route_history]).mode().iloc[0]
        }

    def export_history_to_csv(self, filename: str = "route_history.csv") -> None:
        """
        Exporta el historial de rutas a un archivo CSV.
        
        Args:
            filename: Nombre del archivo CSV
        """
        if not self.route_history:
            logging.warning("No hay historial para exportar")
            return
        
        try:
            # Convertir el historial a DataFrame
            df = pd.DataFrame(self.route_history)
            
            # Expandir la columna de condiciones
            conditions_df = pd.json_normalize(df['conditions'])
            df = pd.concat([df.drop('conditions', axis=1), conditions_df], axis=1)
            
            # Guardar a CSV
            df.to_csv(filename, index=False)
            logging.info(f"Historial exportado a {filename}")
        except Exception as e:
            logging.error(f"Error al exportar historial: {str(e)}")


def main():
    try:
        route_finder = RouteFinderML()
        
        # Mostrar estadísticas si hay historial
        if route_finder.get_route_history():
            stats = route_finder.get_route_statistics()
            print("\nEstadísticas de búsquedas anteriores:")
            print(f"  Total de rutas buscadas: {stats['total_routes']}")
            print(f"  Costo promedio: ${stats['average_cost']:.2f}")
            print(f"  Ruta más económica: ${stats['min_cost']:.2f}")
            print(f"  Ruta más costosa: ${stats['max_cost']:.2f}")
            print(f"  Origen más común: {stats['most_common_origin']}")
            print(f"  Destino más común: {stats['most_common_destination']}")
        
        while True:
            print("\n=== SISTEMA DE BÚSQUEDA DE RUTAS ÓPTIMAS ===")
            print("1. Buscar nueva ruta")
            print("2. Ver historial de rutas")
            print("3. Exportar historial a CSV")
            print("4. Cambiar condiciones de búsqueda")
            print("5. Salir")
            
            opcion = input("\nSeleccione una opción (1-5): ")
            
            if opcion == "1":
                origen = input("Ingrese el punto de partida: ").lower()
                destino = input("Ingrese el punto de destino: ").lower()
                
                mejor_camino, mejor_costo = route_finder.search_best_route_ml(origen, destino)
                
                if mejor_camino:
                    print(f"\nLa mejor ruta de ({origen}) hacia ({destino}) es:")
                    print(" -> ".join(mejor_camino))
                    print(f"Costo total estimado: ${mejor_costo:.2f}")
                else:
                    print(f"\nNo se encontró una ruta válida de ({origen}) hacia ({destino}).")
            
            elif opcion == "2":
                history = route_finder.get_route_history()
                if not history:
                    print("\nNo hay historial de rutas.")
                    continue
                
                print("\nHistorial de rutas:")
                for i, route in enumerate(history[-10:], 1):  # Mostrar las últimas 10 rutas
                    print(f"{i}. {route['origin']} -> {route['destination']} (${route['cost']:.2f})")
            
            elif opcion == "3":
                filename = input("Ingrese el nombre del archivo (default: route_history.csv): ") or "route_history.csv"
                route_finder.export_history_to_csv(filename)
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
        logging.error(f"Error en la ejecución principal: {str(e)}")
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
