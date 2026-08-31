"""
Módulo de Configuración: Optimización de 6 Focos en Habitación de 56 m²
Basado en los conceptos de la Clase 4 - Búsquedas Locales (IA)
"""

from dataclasses import dataclass
from typing import Tuple

# ==============================================================================
# 1. GEOMETRÍA DE LA HABITACIÓN (56 m²)
# ==============================================================================
ROOM_WIDTH = 7.0       # Ancho en metros (X)
ROOM_LENGTH = 8.0      # Largo en metros (Y)
ROOM_AREA = ROOM_WIDTH * ROOM_LENGTH  # 7m * 8m = 56 m²
ROOM_HEIGHT = 2.8      # Altura del techo en metros
WORKPLANE_HEIGHT = 0.75 # Altura del plano de trabajo en metros (altura estándar mesa)
EFFECTIVE_HEIGHT = ROOM_HEIGHT - WORKPLANE_HEIGHT  # 2.05 m de distancia vertical

# Margen mínimo de separación con las paredes para colocar los focos (en metros)
MARGIN_WALL = 0.5

# Discretización de la habitación para muestreo de iluminancia
GRID_NX = 14  # Puntos de muestreo en X (cada 0.5 m)
GRID_NY = 16  # Puntos de muestreo en Y (cada 0.5 m)
TOTAL_SAMPLE_POINTS = GRID_NX * GRID_NY  # 224 puntos de evaluación en el cuarto de 56m²

# ==============================================================================
# 2. ESPECIFICACIONES DE LOS 6 FOCOS
# ==============================================================================
NUM_BULBS = 6

@dataclass(frozen=True)
class BulbModel:
    name: str
    wattage: float       # Potencia eléctrica en Watts
    lumens: float        # Flujo luminoso total en lúmenes (lm)
    luminous_intensity: float  # Intensidad luminosa central I (Candelas)

# Catálogo de focos LED comerciales disponibles
BULB_CATALOG: Tuple[BulbModel, ...] = (
    BulbModel(name="Apagado (0W)",   wattage=0.0,  lumens=0.0,    luminous_intensity=0.0),
    BulbModel(name="LED 9W",        wattage=9.0,  lumens=800.0,  luminous_intensity=254.6),
    BulbModel(name="LED 12W",       wattage=12.0, lumens=1100.0, luminous_intensity=350.1),
    BulbModel(name="LED 15W",       wattage=15.0, lumens=1500.0, luminous_intensity=477.5),
    BulbModel(name="LED 20W",       wattage=20.0, lumens=2000.0, luminous_intensity=636.6),
    BulbModel(name="LED 30W",       wattage=30.0, lumens=3000.0, luminous_intensity=954.9),
)

# ==============================================================================
# 3. NORMAS DE ILUMINACIÓN Y PARÁMETROS ECONÓMICOS
# ==============================================================================
# Norma estándar para habitación de estudio/vivienda/oficina (UNE-EN 12464-1)
TARGET_LUX = 300.0          # Iluminancia media recomendada (300 Lux)
MIN_ACCEPTABLE_LUX = 250.0  # Mínimo aceptable para evitar fatiga visual
TARGET_UNIFORMITY = 0.60    # Relación Emin / Eavg deseada (>= 0.60 para confort visual)

# Parámetros económicos de costo
ELECTRICITY_RATE_KWH = 0.12  # Tarifa eléctrica en USD por kWh
USAGE_HOURS_PER_DAY = 8.0    # Horas de uso diario estimadas
DAYS_PER_MONTH = 30.0        # Días de facturación mensual

# ==============================================================================
# 4. PARÁMETROS DEL TEMPLE SIMULADO (SIMULATED ANNEALING - CLASE 4)
# ==============================================================================
# La función de costo h(s) varía típicamente entre 0.2 y 1.0. 
# T0 = 1.5 garantiza una aceptación inicial de movimientos peores del ~80-90%.
# Tmin = 0.0001 asegura congelamiento térmico (explotación pura) al finalizar.
SA_INITIAL_TEMP = 1.5        # Temperatura inicial T0
SA_FINAL_TEMP = 0.0001       # Temperatura mínima de parada Tmin
SA_COOLING_RATE = 0.95       # Factor de enfriamiento alfa (Tk+1 = alfa * Tk)
SA_STEPS_PER_TEMP = 35       # Iteraciones (épocas internas) por nivel de temperatura
SA_PERTURBATION_STEP = 0.60  # Desplazamiento máximo aleatorio en coordenadas (metros)

# ==============================================================================
# 5. PONDERACIONES DE LA FUNCIÓN HEURÍSTICA DE COSTO h(s)
# ==============================================================================
# La función objetivo equilibra la calidad lumínica y el ahorro de energía:
WEIGHT_LUX_DEFICIT = 0.45    # Penalización por no alcanzar los 300 Lux objetivo
WEIGHT_UNIFORMITY = 0.30     # Penalización por zonas oscuras / falta de uniformidad
WEIGHT_POWER_COST = 0.25     # Penalización por consumo excesivo de potencia (Watts)
