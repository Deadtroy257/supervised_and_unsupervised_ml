# Definición del grafo de transporte
TRANSPORT_GRAPH = {
    "nodes": ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
    "connections": [
        ('a', 'b', 10),
        ('a', 'c', 15),
        ('b', 'd', 12),
        ('c', 'd', 10),
        ('c', 'e', 20),
        ('d', 'e', 8),
        ('d', 'f', 15),
        ('e', 'f', 5),
        ('e', 'g', 12),
        ('f', 'g', 8),
        ('f', 'h', 10),
        ('g', 'h', 6),
        ('g', 'i', 15),
        ('h', 'i', 8),
        ('h', 'j', 12),
        ('i', 'j', 10)
    ]
}

# Base de conocimiento con condiciones globales
KNOWLEDGE_BASE = {
    "hora_pico": False,
    "feriado": True,
    "tipo_vehiculo": "autobus",
    "trafico": "medio"
}