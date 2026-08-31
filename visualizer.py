"""
Módulo de Visualización y Gráficos
Genera mapas de calor de iluminancia 2D en la habitación de 56 m²,
curvas de convergencia de Temple Simulado y gráficos comparativos de KPIs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from config import (
    ROOM_WIDTH, ROOM_LENGTH, TARGET_LUX, BULB_CATALOG
)
from lighting_model import get_grid_coordinates, calculate_illuminance_matrix
from state import RoomLightingState
from objective import LightingKPIs
from simulated_annealing import IterationLog, SAResult


def ensure_output_dir(output_dir: str = "output") -> str:
    """Crea el directorio de salida si no existe."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_comparative_heatmaps(
    initial_state: RoomLightingState,
    best_state: RoomLightingState,
    initial_kpis: LightingKPIs,
    best_kpis: LightingKPIs,
    output_path: str = "output/mapa_iluminacion_comparativo.png"
) -> str:
    """
    Genera un gráfico comparativo de 2 paneles con los mapas de calor de iluminancia (Lux)
    y la posición de los 6 focos antes y después de la optimización con Temple Simulado.
    """
    ensure_output_dir(os.path.dirname(output_path) or "output")
    x, y, X, Y = get_grid_coordinates()

    E_init = calculate_illuminance_matrix(
        initial_state.get_positions_array(),
        initial_state.get_model_indices_array()
    )
    E_best = calculate_illuminance_matrix(
        best_state.get_positions_array(),
        best_state.get_model_indices_array()
    )

    vmin = 0.0
    vmax = max(500.0, float(np.max([E_init, E_best])))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.patch.set_facecolor('#0f172a') # Dark modern theme

    configs = [
        (axes[0], E_init, initial_state, initial_kpis, "1. Estado Inicial (Distribución Aleatoria)", "#ef4444"),
        (axes[1], E_best, best_state, best_kpis, "2. Estado Optimizado (Temple Simulado - Clase 4)", "#10b981")
    ]

    for ax, E_mat, state, kpis, title, border_col in configs:
        ax.set_facecolor('#1e293b')
        
        # Mapa de calor con isolíneas
        cf = ax.contourf(X, Y, E_mat, levels=30, cmap='plasma', vmin=vmin, vmax=vmax)
        cs = ax.contour(X, Y, E_mat, levels=[200, 270, 300, 350, 400], colors='#ffffff', alpha=0.4, linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f lx', colors='#ffffff')

        # Dibujar los 6 focos
        for i, b in enumerate(state.bulbs):
            model = BULB_CATALOG[b.model_idx]
            if b.model_idx > 0:
                # Foco encendido
                ax.scatter(b.x, b.y, color='#fbbf24', s=160, edgecolors='#ffffff', linewidth=1.5, zorder=5)
                ax.annotate(
                    f"F{i+1}: {model.name}\n({b.x:.1f}, {b.y:.1f})m",
                    (b.x, b.y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=7.5,
                    fontweight='bold',
                    color='#ffffff',
                    bbox=dict(boxstyle="round,pad=0.2", fc="#000000", ec=border_col, alpha=0.8, lw=1)
                )
            else:
                # Foco apagado
                ax.scatter(b.x, b.y, color='#64748b', s=100, marker='x', linewidth=2, zorder=5)
                ax.annotate(
                    f"F{i+1}: Off",
                    (b.x, b.y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha='center',
                    fontsize=7,
                    color='#94a3b8',
                    bbox=dict(boxstyle="round,pad=0.1", fc="#0f172a", ec="#64748b", alpha=0.7)
                )

        # Límites del cuarto de 56 m² (7m x 8m)
        ax.set_xlim(0, ROOM_WIDTH)
        ax.set_ylim(0, ROOM_LENGTH)
        ax.set_xlabel("Ancho de la habitación (metros)", color='#e2e8f0', fontsize=10, labelpad=6)
        ax.set_ylabel("Largo de la habitación (metros)", color='#e2e8f0', fontsize=10, labelpad=6)
        ax.set_title(title, color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#475569')

        # Cuadro de KPIs resumen
        kpi_text = (
            f"Lux Promedio: {kpis.avg_lux:.1f} lx (Meta: {TARGET_LUX:.0f})\n"
            f"Uniformidad: {kpis.uniformity:.2f} (Emin/Eavg)\n"
            f"Cobertura Norma: {kpis.coverage_pct:.1f}%\n"
            f"Potencia: {kpis.total_watts:.1f} W ({kpis.active_bulbs}/6 focos)\n"
            f"Costo Eléctrico: ${kpis.monthly_cost_usd:.2f}/mes\n"
            f"Costo Heurístico h(s): {kpis.heuristic_cost:.4f}"
        )
        ax.text(
            0.03, 0.04, kpi_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment='bottom',
            color='#f8fafc',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#090d16', edgecolor=border_col, alpha=0.85, lw=1.2)
        )

    fig.suptitle(
        f"Optimización de Iluminación y Costo: 6 Focos en Habitación de {ROOM_WIDTH*ROOM_LENGTH:.0f} m² (7m × 8m)",
        fontsize=14, fontweight='bold', color='#f8fafc', y=0.98
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.90, 0.94])

    # Barra de color común
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Iluminancia en el Plano de Trabajo (Lux)', color='#f8fafc', fontsize=10, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='#f8fafc')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e2e8f0')

    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return output_path


def plot_sa_convergence(
    history: List[IterationLog],
    output_path: str = "output/curva_temple_simulado.png"
) -> str:
    """
    Genera el gráfico de convergencia del algoritmo de Temple Simulado:
    - Descenso de Temperatura T vs Épocas.
    - Evolución de la Función Heurística de Costo h(s) (Actual vs Mejor).
    - Tasa de aceptación de movimientos.
    """
    ensure_output_dir(os.path.dirname(output_path) or "output")

    epochs = [h.epoch for h in history]
    temperatures = [h.temperature for h in history]
    current_costs = [h.current_cost for h in history]
    best_costs = [h.best_cost for h in history]
    acc_rates = [h.acceptance_rate for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=150, sharex=True)
    fig.patch.set_facecolor('#0f172a')

    # Subplot 1: Función de Costo h(s) y Temperatura
    ax1.set_facecolor('#1e293b')
    line1 = ax1.plot(epochs, current_costs, color='#38bdf8', alpha=0.5, label='Costo h(s) Estado Actual', linewidth=1.2)
    line2 = ax1.plot(epochs, best_costs, color='#10b981', linewidth=2.5, label='Mejor Costo h(s) Encontrado')
    ax1.set_ylabel("Costo Heurístico h(s)", color='#f8fafc', fontsize=10, labelpad=8)
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(True, linestyle='--', alpha=0.2, color='#64748b')
    for spine in ax1.spines.values():
        spine.set_color('#475569')

    # Eje secundario para la Temperatura
    ax1_t = ax1.twinx()
    line3 = ax1_t.plot(epochs, temperatures, color='#f59e0b', linestyle=':', linewidth=2, label='Temperatura T')
    ax1_t.set_ylabel("Temperatura T", color='#f59e0b', fontsize=10, labelpad=8)
    ax1_t.tick_params(colors='#f59e0b')
    ax1_t.spines['right'].set_color('#f59e0b')

    # Leyenda combinada
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', facecolor='#090d16', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)
    ax1.set_title("Convergencia del Temple Simulado: Reducción del Costo h(s) y Enfriamiento", color='#f8fafc', fontsize=12, fontweight='bold')

    # Subplot 2: Tasa de Aceptación (%)
    ax2.set_facecolor('#1e293b')
    ax2.plot(epochs, acc_rates, color='#ec4899', linewidth=1.8, label='Tasa de Aceptación (%)')
    ax2.fill_between(epochs, 0, acc_rates, color='#ec4899', alpha=0.15)
    ax2.set_xlabel("Épocas de Temperatura (Niveles de Enfriamiento)", color='#f8fafc', fontsize=10, labelpad=8)
    ax2.set_ylabel("Aceptación (%)", color='#ec4899', fontsize=10, labelpad=8)
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(True, linestyle='--', alpha=0.2, color='#64748b')
    for spine in ax2.spines.values():
        spine.set_color('#475569')
    ax2.legend(loc='upper right', facecolor='#090d16', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)
    ax2.set_title("Dinámica de Exploración vs Explotación: Tasa de Aceptación de Metrópolis", color='#f8fafc', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return output_path


def plot_kpi_comparison_bars(
    initial_kpis: LightingKPIs,
    best_kpis: LightingKPIs,
    output_path: str = "output/comparacion_kpis.png"
) -> str:
    """
    Genera un gráfico de barras comparativo de los KPIs clave entre el estado inicial y el óptimo.
    """
    ensure_output_dir(os.path.dirname(output_path) or "output")

    categories = [
        "Lux Promedio\n(lx)",
        "Cobertura\n(%)",
        "Uniformidad\n(x100)",
        "Potencia Total\n(W)",
        "Costo Mensual\n($ USD)",
        "Costo Heurístico\nh(s)"
    ]

    init_vals = [
        initial_kpis.avg_lux,
        initial_kpis.coverage_pct,
        initial_kpis.uniformity * 100.0,
        initial_kpis.total_watts,
        initial_kpis.monthly_cost_usd,
        initial_kpis.heuristic_cost
    ]

    best_vals = [
        best_kpis.avg_lux,
        best_kpis.coverage_pct,
        best_kpis.uniformity * 100.0,
        best_kpis.total_watts,
        best_kpis.monthly_cost_usd,
        best_kpis.heuristic_cost
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    rects1 = ax.bar(x - width/2, init_vals, width, label='Estado Inicial (Aleatorio)', color='#ef4444', edgecolor='#fca5a5', alpha=0.85)
    rects2 = ax.bar(x + width/2, best_vals, width, label='Estado Optimizado (Temple Simulado)', color='#10b981', edgecolor='#6ee7b7', alpha=0.90)

    # Etiquetas de valor en cada barra
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, color='#f8fafc', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    ax.set_ylabel('Valor de la Métrica', color='#f8fafc', fontsize=11)
    ax.set_title('Comparativa de Rendimiento y KPIs: Inicial vs Optimizado (Clase 4)', color='#f8fafc', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color='#e2e8f0', fontsize=9.5)
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#090d16', edgecolor='#475569', labelcolor='#f8fafc', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.15, axis='y', color='#64748b')
    for spine in ax.spines.values():
        spine.set_color('#475569')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    return output_path
