"""
Módulo de Función Objetivo y Factores Medibles (KPIs)
Implementa los Factores Medibles (KPIs fotométricos y económicos) y la Función
Heurística de Costo h(s) para la Búsqueda Local con Temple Simulado (Clase 4).
"""

import numpy as np
from dataclasses import dataclass
from config import (
    TARGET_LUX, MIN_ACCEPTABLE_LUX, TARGET_UNIFORMITY,
    NUM_BULBS, BULB_CATALOG,
    ELECTRICITY_RATE_KWH, USAGE_HOURS_PER_DAY, DAYS_PER_MONTH,
    WEIGHT_LUX_DEFICIT, WEIGHT_UNIFORMITY, WEIGHT_POWER_COST
)
from lighting_model import calculate_illuminance_matrix
from state import RoomLightingState


@dataclass
class LightingKPIs:
    """Contenedor estructurado de los Indicadores Clave de Rendimiento (KPIs)."""
    avg_lux: float            # Iluminancia promedio en la habitación (Lux)
    min_lux: float            # Iluminancia mínima en la zona más oscura (Lux)
    max_lux: float            # Iluminancia máxima en la zona más brillante (Lux)
    uniformity: float         # Uniformidad Emin / Eavg (0.0 a 1.0)
    coverage_pct: float       # Porcentaje del área de 56 m² que cumple la norma (>= 270 Lux)
    total_watts: float        # Potencia eléctrica total consumida (Watts)
    active_bulbs: int         # Número de focos encendidos (de 6)
    monthly_cost_usd: float   # Costo mensual estimado de energía eléctrica (USD)
    heuristic_cost: float     # Valor de costo heurístico h(s) [A MINIMIZAR en Temple Simulado]


def evaluate_kpis(state: RoomLightingState) -> LightingKPIs:
    """
    Evalúa de forma exhaustiva el estado de iluminación de la habitación de 56 m²
    y calcula todos los KPIs fotométricos, económicos y el costo heurístico.
    """
    positions = state.get_positions_array()
    model_indices = state.get_model_indices_array()
    
    # Matriz de iluminancia 2D en toda la habitación
    E_matrix = calculate_illuminance_matrix(positions, model_indices)
    
    avg_lux = float(np.mean(E_matrix))
    min_lux = float(np.min(E_matrix))
    max_lux = float(np.max(E_matrix))
    
    # Uniformidad U0 = Emin / Eavg
    uniformity = (min_lux / avg_lux) if avg_lux > 1e-6 else 0.0
    uniformity = float(np.clip(uniformity, 0.0, 1.0))
    
    # Cobertura: Porcentaje de puntos que cumplen al menos el 90% del objetivo (270 Lux)
    threshold_lux = 0.90 * TARGET_LUX
    coverage_pct = float(np.mean(E_matrix >= threshold_lux) * 100.0)
    
    # Potencia y Costos
    total_watts = state.get_total_wattage()
    active_bulbs = state.get_active_bulbs_count()
    
    # Consumo mensual = (Watts / 1000) * Horas/día * Días/mes * Tarifa USD/kWh
    monthly_kwh = (total_watts / 1000.0) * USAGE_HOURS_PER_DAY * DAYS_PER_MONTH
    monthly_cost_usd = float(monthly_kwh * ELECTRICITY_RATE_KWH)
    
    # --------------------------------------------------------------------------
    # CÁLCULO DE LA FUNCIÓN HEURÍSTICA DE COSTO h(s)
    # --------------------------------------------------------------------------
    # 1. Penalización por déficit o exceso de iluminación relativa a 300 Lux
    if avg_lux < TARGET_LUX:
        # Penalización cuadrática si falta luz
        lux_penalty = ((TARGET_LUX - avg_lux) / TARGET_LUX) ** 2
    else:
        # Penalización suave si hay sobreiluminación excesiva (> 450 Lux)
        lux_penalty = max(0.0, (avg_lux - 450.0) / 300.0)
        
    # 2. Penalización por falta de uniformidad y zonas oscuras
    dark_spots_ratio = np.mean(E_matrix < MIN_ACCEPTABLE_LUX)
    uniformity_penalty = (1.0 - uniformity) + float(dark_spots_ratio)
    
    # 3. Penalización por potencia consumida normalizada (Máx posible = 6 focos * 30W = 180W)
    max_possible_watts = NUM_BULBS * 30.0
    power_penalty = (total_watts / max_possible_watts)
    
    # Costo Heurístico combinado h(s) (>= 0, a minimizar)
    heuristic_cost = float(
        WEIGHT_LUX_DEFICIT * lux_penalty +
        WEIGHT_UNIFORMITY * uniformity_penalty +
        WEIGHT_POWER_COST * power_penalty
    )
    
    return LightingKPIs(
        avg_lux=avg_lux,
        min_lux=min_lux,
        max_lux=max_lux,
        uniformity=uniformity,
        coverage_pct=coverage_pct,
        total_watts=total_watts,
        active_bulbs=active_bulbs,
        monthly_cost_usd=monthly_cost_usd,
        heuristic_cost=heuristic_cost
    )


def cost_function(state: RoomLightingState) -> float:
    """Función de costo heurístico h(s) para Búsqueda Local (Temple Simulado)."""
    return evaluate_kpis(state).heuristic_cost

