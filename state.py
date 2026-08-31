"""
Módulo de Representación del Espacio de Estados
Modela la configuración y disposición de los 6 focos en la habitación de 56 m².
Diseñado para la formulación y ejecución de algoritmos de Búsqueda Local (Temple Simulado).
"""

from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from config import (
    NUM_BULBS, ROOM_WIDTH, ROOM_LENGTH, MARGIN_WALL,
    BULB_CATALOG, SA_PERTURBATION_STEP
)


@dataclass
class BulbState:
    """Representa el estado individual de un foco (Posición y Vatiaje)."""
    x: float            # Posición en metros [MARGIN_WALL, ROOM_WIDTH - MARGIN_WALL]
    y: float            # Posición en metros [MARGIN_WALL, ROOM_LENGTH - MARGIN_WALL]
    model_idx: int      # Índice del catálogo de focos (0=Apagado, 1=9W, ..., 5=30W)

    def clamp_bounds(self) -> None:
        """Asegura que el foco permanezca dentro de los límites físicos de la habitación."""
        self.x = float(np.clip(self.x, MARGIN_WALL, ROOM_WIDTH - MARGIN_WALL))
        self.y = float(np.clip(self.y, MARGIN_WALL, ROOM_LENGTH - MARGIN_WALL))
        self.model_idx = int(np.clip(self.model_idx, 0, len(BULB_CATALOG) - 1))


class RoomLightingState:
    """
    Representa el estado global de iluminación de la habitación con los 6 focos.
    Encapsula las coordenadas de cada foco, su modelo de potencia y operaciones de vecindario.
    """

    def __init__(self, bulbs: List[BulbState]):
        if len(bulbs) != NUM_BULBS:
            raise ValueError(f"El estado debe contener exactamente {NUM_BULBS} focos.")
        self.bulbs = bulbs
        for b in self.bulbs:
            b.clamp_bounds()

    @classmethod
    def create_random(cls, rng: np.random.Generator = None) -> RoomLightingState:
        """
        Crea un estado inicial aleatorio de los 6 focos distribuidos en la habitación de 56 m²,
        cumpliendo con la premisa inicial del problema.
        """
        if rng is None:
            rng = np.random.default_rng()

        bulbs = []
        for _ in range(NUM_BULBS):
            x = rng.uniform(MARGIN_WALL, ROOM_WIDTH - MARGIN_WALL)
            y = rng.uniform(MARGIN_WALL, ROOM_LENGTH - MARGIN_WALL)
            # Selección aleatoria de modelo de foco (de 0W a 30W)
            model_idx = int(rng.integers(0, len(BULB_CATALOG)))
            bulbs.append(BulbState(x=x, y=y, model_idx=model_idx))

        return cls(bulbs)

    # --------------------------------------------------------------------------
    # MATRICES NUMPY PARA CÁLCULO VECTORIAL RÁPIDO
    # --------------------------------------------------------------------------
    def get_positions_array(self) -> np.ndarray:
        """Retorna un array de shape (NUM_BULBS, 2) con las coordenadas (x, y)."""
        return np.array([[b.x, b.y] for b in self.bulbs], dtype=np.float64)

    def get_model_indices_array(self) -> np.ndarray:
        """Retorna un array de shape (NUM_BULBS,) con los índices de catálogo."""
        return np.array([b.model_idx for b in self.bulbs], dtype=np.int32)

    def get_total_wattage(self) -> float:
        """Retorna la suma total de potencia en Watts de los focos encendidos."""
        return float(sum(BULB_CATALOG[b.model_idx].wattage for b in self.bulbs))

    def get_active_bulbs_count(self) -> int:
        """Retorna la cantidad de focos actualmente encendidos (con vatiaje > 0W)."""
        return sum(1 for b in self.bulbs if b.model_idx > 0)

    # --------------------------------------------------------------------------
    # SERIALIZACIÓN Y DESERIALIZACIÓN VECTORIAL DEL ESTADO
    # --------------------------------------------------------------------------
    def to_vector(self) -> np.ndarray:
        """
        Codifica el estado en un vector numérico 1D.
        Formato: [x1, y1, idx1, x2, y2, idx2, ..., x6, y6, idx6] -> Longitud = 18
        """
        data = []
        for b in self.bulbs:
            data.extend([b.x, b.y, float(b.model_idx)])
        return np.array(data, dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> RoomLightingState:
        """
        Decodifica un vector numérico 1D en una instancia de RoomLightingState.
        """
        if len(vector) != NUM_BULBS * 3:
            raise ValueError(f"Longitud de vector inválida: {len(vector)}. Se esperaba {NUM_BULBS * 3}.")

        bulbs = []
        for i in range(NUM_BULBS):
            base = i * 3
            x = float(vector[base])
            y = float(vector[base + 1])
            model_idx = int(np.round(vector[base + 2]))
            bulbs.append(BulbState(x=x, y=y, model_idx=model_idx))

        return cls(bulbs)

    # --------------------------------------------------------------------------
    # GENERACIÓN DE VECINDARIO (BÚSQUEDA LOCAL)
    # --------------------------------------------------------------------------
    def get_neighbor(self, perturbation_step: float = SA_PERTURBATION_STEP, rng: np.random.Generator = None) -> RoomLightingState:
        """
        Genera un estado vecino estocástico mediante pequeñas perturbaciones.
        
        Tipos de perturbación aleatoria:
        1. Desplazamiento espacial en (x, y) de uno o varios focos.
        2. Cambio discreto de vatiaje/modelo de un foco (+1, -1 en catálogo).
        3. Encendido / Apagado directo de un foco.
        """
        if rng is None:
            rng = np.random.default_rng()

        new_bulbs = [BulbState(x=b.x, y=b.y, model_idx=b.model_idx) for b in self.bulbs]
        
        # Seleccionar aleatoriamente entre 1 y 2 focos para perturbar
        target_indices = rng.choice(NUM_BULBS, size=rng.integers(1, 3), replace=False)

        for idx in target_indices:
            action_type = rng.random()

            if action_type < 0.60:
                # Perturbación espacial continua (mover el foco ligeramente en el techo)
                dx = rng.uniform(-perturbation_step, perturbation_step)
                dy = rng.uniform(-perturbation_step, perturbation_step)
                new_bulbs[idx].x += dx
                new_bulbs[idx].y += dy
            elif action_type < 0.85:
                # Modificar vatiaje (subir o bajar un nivel en el catálogo comercial)
                delta_idx = rng.choice([-1, 1])
                new_bulbs[idx].model_idx += delta_idx
            else:
                # Encender/apagar o reasignar aleatoriamente el modelo
                if new_bulbs[idx].model_idx > 0 and rng.random() < 0.5:
                    new_bulbs[idx].model_idx = 0  # Apagar para ahorrar costo
                else:
                    new_bulbs[idx].model_idx = int(rng.integers(1, len(BULB_CATALOG)))

            new_bulbs[idx].clamp_bounds()

        return RoomLightingState(new_bulbs)

    def clone(self) -> RoomLightingState:
        """Retorna una copia profunda del estado actual."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        bulbs_str = ", ".join([f"Foco {i+1}: ({b.x:.2f}m, {b.y:.2f}m, {BULB_CATALOG[b.model_idx].name})" for i, b in enumerate(self.bulbs)])
        return f"RoomLightingState({bulbs_str})"

