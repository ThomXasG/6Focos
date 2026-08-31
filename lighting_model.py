"""
Módulo de Modelo Fotométrico: Cálculo de Iluminación en la Habitación de 56 m²
Aplica la Ley Inversa del Cuadrado y la Ley del Coseno de Lambert.
"""

import numpy as np
from typing import List, Tuple
from config import (
    ROOM_WIDTH, ROOM_LENGTH, EFFECTIVE_HEIGHT,
    GRID_NX, GRID_NY, BULB_CATALOG
)


def get_grid_coordinates() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera las coordenadas de muestreo de la habitación discretizada en una malla 2D.
    Retorna (x_coords_1d, y_coords_1d, X_grid_2d, Y_grid_2d).
    """
    x = np.linspace(0.25, ROOM_WIDTH - 0.25, GRID_NX)
    y = np.linspace(0.25, ROOM_LENGTH - 0.25, GRID_NY)
    X, Y = np.meshgrid(x, y)  # Shape: (GRID_NY, GRID_NX)
    return x, y, X, Y


def calculate_illuminance_matrix(bulbs_positions: np.ndarray, bulb_catalog_indices: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de iluminancia total (en Lux) sobre el plano de trabajo de la habitación de 56 m².
    
    Parámetros:
        bulbs_positions: np.ndarray de shape (N, 2) con las coordenadas (x, y) de cada foco.
        bulb_catalog_indices: np.ndarray de shape (N,) con los índices del catálogo para cada foco.
        
    Retorna:
        E_total: np.ndarray de shape (GRID_NY, GRID_NX) con los Luxes en cada punto de la malla.
    """
    _, _, X, Y = get_grid_coordinates()
    E_total = np.zeros_like(X, dtype=np.float64)
    h = EFFECTIVE_HEIGHT

    for i in range(len(bulb_catalog_indices)):
        idx = int(bulb_catalog_indices[i])
        model = BULB_CATALOG[idx]
        
        # Si el foco está apagado (0W / 0 lm), no aporta iluminancia
        if model.luminous_intensity <= 0.0:
            continue
            
        bx, by = bulbs_positions[i, 0], bulbs_positions[i, 1]
        
        # Distancia en el plano horizontal (dx, dy)
        dx = X - bx
        dy = Y - by
        
        # Distancia 3D euclidiana al punto del plano de trabajo: d = sqrt(dx^2 + dy^2 + h^2)
        d_sq = dx**2 + dy**2 + h**2
        d = np.sqrt(d_sq)
        
        # Ley de Lambert: cos(theta) = h / d
        # Iluminancia puntual: E = (I * cos(theta)) / d^2 = (I * h) / (d^3)
        E_bulb = (model.luminous_intensity * h) / (d_sq * d)
        
        E_total += E_bulb

    return E_total
