"""
Programa Principal: Optimización de Iluminación y Costo (6 Focos en 56 m²)
Deber de Inteligencia Artificial - Séptimo Semestre
Algoritmo de Búsqueda Local: Temple Simulado (Simulated Annealing - Clase 4).
"""

import os
import sys
from config import (
    ROOM_WIDTH, ROOM_LENGTH, ROOM_AREA, ROOM_HEIGHT, TARGET_LUX,
    NUM_BULBS, BULB_CATALOG, SA_INITIAL_TEMP, SA_FINAL_TEMP, SA_COOLING_RATE,
    SA_STEPS_PER_TEMP
)
from state import RoomLightingState
from objective import evaluate_kpis
from simulated_annealing import SimulatedAnnealingOptimizer
from visualizer import (
    plot_comparative_heatmaps,
    plot_sa_convergence,
    plot_kpi_comparison_bars,
    ensure_output_dir
)


def print_banner():
    print("=" * 80)
    print("  DEBER DE INTELIGENCIA ARTIFICIAL - OPTIMIZACIÓN DE ILUMINACIÓN Y COSTO")
    print("  Algoritmo de Búsqueda Local: Temple Simulado (Simulated Annealing - Clase 4)")
    print("  Espacio: Habitación de 56 m² (7.0m x 8.0m, Altura: 2.8m) | Focos: 6 unidades")
    print("=" * 80)


def print_bulbs_table(title: str, state: RoomLightingState):
    print(f"\n--- {title} ---")
    print(f"{'Foco':<6} | {'Posición (X, Y)':<18} | {'Modelo LED':<16} | {'Potencia (W)':<12} | {'Lúmenes (lm)':<12}")
    print("-" * 75)
    for i, b in enumerate(state.bulbs):
        model = BULB_CATALOG[b.model_idx]
        pos_str = f"({b.x:.2f} m, {b.y:.2f} m)"
        print(f"Foco {i+1:<2} | {pos_str:<18} | {model.name:<16} | {model.wattage:<12.1f} | {model.lumens:<12.1f}")
    print("-" * 75)
    print(f"Total Potencia: {state.get_total_wattage():.1f} W | Focos Encendidos: {state.get_active_bulbs_count()}/{NUM_BULBS}")


def print_kpi_comparison(init_kpis, opt_kpis):
    print("\n" + "=" * 80)
    print(f"{'MÉTRICA / KPI':<32} | {'INICIAL (Aleatorio)':<20} | {'OPTIMIZADO (Temple)':<20} | {'VARIACIÓN':<10}")
    print("=" * 80)
    
    rows = [
        ("Iluminancia Media (Lux)", f"{init_kpis.avg_lux:.1f} lx", f"{opt_kpis.avg_lux:.1f} lx", f"{opt_kpis.avg_lux - init_kpis.avg_lux:+.1f} lx"),
        ("Iluminancia Mínima (Lux)", f"{init_kpis.min_lux:.1f} lx", f"{opt_kpis.min_lux:.1f} lx", f"{opt_kpis.min_lux - init_kpis.min_lux:+.1f} lx"),
        ("Uniformidad (Emin / Eavg)", f"{init_kpis.uniformity:.3f}", f"{opt_kpis.uniformity:.3f}", f"{opt_kpis.uniformity - init_kpis.uniformity:+.3f}"),
        ("Cobertura de Norma (>=270 lx)", f"{init_kpis.coverage_pct:.1f} %", f"{opt_kpis.coverage_pct:.1f} %", f"{opt_kpis.coverage_pct - init_kpis.coverage_pct:+.1f} %"),
        ("Potencia Eléctrica Total", f"{init_kpis.total_watts:.1f} W", f"{opt_kpis.total_watts:.1f} W", f"{opt_kpis.total_watts - init_kpis.total_watts:+.1f} W"),
        ("Focos Encendidos Activos", f"{init_kpis.active_bulbs} / {NUM_BULBS}", f"{opt_kpis.active_bulbs} / {NUM_BULBS}", f"{opt_kpis.active_bulbs - init_kpis.active_bulbs:+d}"),
        ("Costo Estimado Mensual", f"${init_kpis.monthly_cost_usd:.2f} USD", f"${opt_kpis.monthly_cost_usd:.2f} USD", f"${opt_kpis.monthly_cost_usd - init_kpis.monthly_cost_usd:+.2f}"),
        ("Costo Heurístico h(s) [Minimizar]", f"{init_kpis.heuristic_cost:.4f}", f"{opt_kpis.heuristic_cost:.4f}", f"{opt_kpis.heuristic_cost - init_kpis.heuristic_cost:+.4f}")
    ]
    
    for label, init_val, opt_val, diff_val in rows:
        print(f"{label:<32} | {init_val:<20} | {opt_val:<20} | {diff_val:<10}")
    print("=" * 80)


def main():
    print_banner()
    ensure_output_dir("output")

    # 1. Crear Estado Inicial Aleatorio (6 focos distribuidos aleatoriamente en 56 m²)
    print("\n[PASO 1] Generando distribución aleatoria inicial de 6 focos en la habitación...")
    initial_state = RoomLightingState.create_random(rng=None)
    init_kpis = evaluate_kpis(initial_state)
    print_bulbs_table("ESTADO INICIAL (DISTRIBUCIÓN ALEATORIA)", initial_state)

    # 2. Ejecutar Algoritmo de Búsqueda Local: Temple Simulado (Clase 4)
    print("\n[PASO 2] Ejecutando Búsqueda Local por Temple Simulado (Simulated Annealing)...")
    print(f"Parámetros: T0={SA_INITIAL_TEMP}, Tmin={SA_FINAL_TEMP}, alfa={SA_COOLING_RATE}, pasos/temp={SA_STEPS_PER_TEMP}")
    
    optimizer = SimulatedAnnealingOptimizer(
        initial_temp=SA_INITIAL_TEMP,
        final_temp=SA_FINAL_TEMP,
        cooling_rate=SA_COOLING_RATE,
        steps_per_temp=SA_STEPS_PER_TEMP,
        seed=42
    )

    def progress_callback(log):
        if log.epoch % 20 == 0 or log.epoch == 1:
            print(f"  > Época {log.epoch:3d} | Temp: {log.temperature:8.4f} | Costo Actual h(s): {log.current_cost:.4f} | Mejor h(s): {log.best_cost:.4f} | Aceptación: {log.acceptance_rate:5.1f}%")

    result = optimizer.optimize(initial_state=initial_state, callback=progress_callback)
    
    print(f"\n[OK] Optimización finalizada en {result.execution_time_sec:.3f} segundos.")
    print(f"Total de evaluaciones de la función objetivo: {result.total_evaluations}")

    # 3. Mostrar Estado Final Optimizado
    print_bulbs_table("ESTADO OPTIMIZADO (TEMPLE SIMULADO)", result.best_state)

    # 4. Tabla Comparativa de Rendimiento
    print_kpi_comparison(result.initial_kpis, result.best_kpis)

    # 5. Generar Visualizaciones y Gráficos
    print("\n[PASO 3] Generando gráficos de alta resolución en la carpeta 'output/'...")
    p1 = plot_comparative_heatmaps(result.initial_state, result.best_state, result.initial_kpis, result.best_kpis)
    print(f"  [+] Guardado: {p1}")
    p2 = plot_sa_convergence(result.history)
    print(f"  [+] Guardado: {p2}")
    p3 = plot_kpi_comparison_bars(result.initial_kpis, result.best_kpis)
    print(f"  [+] Guardado: {p3}")

    print("\n" + "=" * 80)
    print("  OPTIMIZACIÓN COMPLETADA CON ÉXITO")
    print("=" * 80)


if __name__ == "__main__":
    main()

