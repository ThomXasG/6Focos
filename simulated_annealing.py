"""
Módulo de Búsqueda Local: Algoritmo de Temple Simulado (Simulated Annealing)
Implementación rigurosa basada en los conceptos teóricos de la Clase 4 - Búsquedas Locales.
"""

from __future__ import annotations
import math
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable
from config import (
    SA_INITIAL_TEMP, SA_FINAL_TEMP, SA_COOLING_RATE,
    SA_STEPS_PER_TEMP, SA_PERTURBATION_STEP
)
from state import RoomLightingState
from objective import evaluate_kpis, LightingKPIs


@dataclass
class IterationLog:
    """Registro detallado de una época del Temple Simulado."""
    epoch: int
    temperature: float
    current_cost: float
    best_cost: float
    accepted_moves: int
    worse_accepted_moves: int
    acceptance_rate: float


@dataclass
class SAResult:
    """Resultado final del proceso de optimización por Temple Simulado."""
    initial_state: RoomLightingState
    best_state: RoomLightingState
    initial_kpis: LightingKPIs
    best_kpis: LightingKPIs
    history: List[IterationLog]
    total_iterations: int
    total_evaluations: int
    execution_time_sec: float


class SimulatedAnnealingOptimizer:
    """
    Optimizador de Búsqueda Local mediante Temple Simulado (Simulated Annealing).
    
    Características clave según la Clase 4:
    - Escapa de óptimos locales aceptando probabilísticamente soluciones peores con P = exp(-Delta_h / T).
    - Aplica un esquema de enfriamiento geométrico T = alpha * T.
    - Converge hacia el óptimo global a medida que la temperatura desciende.
    """

    def __init__(
        self,
        initial_temp: float = SA_INITIAL_TEMP,
        final_temp: float = SA_FINAL_TEMP,
        cooling_rate: float = SA_COOLING_RATE,
        steps_per_temp: int = SA_STEPS_PER_TEMP,
        perturbation_step: float = SA_PERTURBATION_STEP,
        seed: Optional[int] = None
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.perturbation_step = perturbation_step
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        initial_state: Optional[RoomLightingState] = None,
        callback: Optional[Callable[[IterationLog], None]] = None
    ) -> SAResult:
        """
        Ejecuta el algoritmo de Temple Simulado para optimizar la iluminación y costo.
        
        Parámetros:
            initial_state: Estado inicial de los 6 focos (si es None, se genera uno aleatorio).
            callback: Función opcional para reportar el progreso en cada época.
            
        Retorna:
            SAResult con el estado óptimo, métricas y traza histórica.
        """
        start_time = time.perf_counter()

        # 1. Estado Inicial Aleatorio (Premisa de la Clase 4)
        if initial_state is None:
            current_state = RoomLightingState.create_random(rng=self.rng)
        else:
            current_state = initial_state.clone()

        initial_state_backup = current_state.clone()
        initial_kpis = evaluate_kpis(initial_state_backup)
        
        current_kpis = initial_kpis
        current_cost = current_kpis.heuristic_cost

        best_state = current_state.clone()
        best_kpis = current_kpis
        best_cost = current_cost

        # 2. Inicialización de Parámetros de Temple
        T = self.initial_temp
        epoch = 0
        total_evaluations = 1
        history: List[IterationLog] = []

        # 3. Bucle Principal de Enfriamiento
        while T > self.final_temp:
            epoch += 1
            accepted_count = 0
            worse_accepted_count = 0

            # Época interna: pasos de búsqueda al nivel de temperatura T actual
            for _ in range(self.steps_per_temp):
                # Generar estado vecino estocástico mediante perturbación local
                neighbor_state = current_state.get_neighbor(
                    perturbation_step=self.perturbation_step,
                    rng=self.rng
                )
                neighbor_kpis = evaluate_kpis(neighbor_state)
                neighbor_cost = neighbor_kpis.heuristic_cost
                total_evaluations += 1

                # Variación de la función heurística de costo: Delta_h = h(vecino) - h(actual)
                delta_h = neighbor_cost - current_cost

                # Criterio de Aceptación de Metrópolis (Clase 4)
                if delta_h < 0.0:
                    # El vecino mejora el costo -> Se acepta incondicionalmente
                    current_state = neighbor_state
                    current_cost = neighbor_cost
                    current_kpis = neighbor_kpis
                    accepted_count += 1

                    # Actualizar el mejor estado global encontrado
                    if current_cost < best_cost:
                        best_state = current_state.clone()
                        best_cost = current_cost
                        best_kpis = current_kpis
                else:
                    # El vecino empeora el costo -> Se calcula la probabilidad de Boltzmann
                    # P = exp(-Delta_h / T)
                    try:
                        prob = math.exp(-delta_h / max(T, 1e-9))
                    except OverflowError:
                        prob = 0.0

                    # Lanzar número aleatorio uniforme r in [0, 1)
                    r = self.rng.random()
                    if r < prob:
                        # Se acepta el movimiento peor para escapar de mínimos locales
                        current_state = neighbor_state
                        current_cost = neighbor_cost
                        current_kpis = neighbor_kpis
                        accepted_count += 1
                        worse_accepted_count += 1

            acceptance_rate = (accepted_count / self.steps_per_temp) * 100.0

            log_entry = IterationLog(
                epoch=epoch,
                temperature=T,
                current_cost=current_cost,
                best_cost=best_cost,
                accepted_moves=accepted_count,
                worse_accepted_moves=worse_accepted_count,
                acceptance_rate=acceptance_rate
            )
            history.append(log_entry)

            if callback:
                callback(log_entry)

            # Enfriamiento geométrico: T_{k+1} = alpha * T_k
            T *= self.cooling_rate

        execution_time = time.perf_counter() - start_time

        return SAResult(
            initial_state=initial_state_backup,
            best_state=best_state,
            initial_kpis=initial_kpis,
            best_kpis=best_kpis,
            history=history,
            total_iterations=epoch * self.steps_per_temp,
            total_evaluations=total_evaluations,
            execution_time_sec=execution_time
        )
