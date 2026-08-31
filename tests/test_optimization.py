import sys
import os
import unittest
import numpy as np

# Asegurar que el directorio raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    ROOM_WIDTH, ROOM_LENGTH, ROOM_AREA, NUM_BULBS, BULB_CATALOG, TARGET_LUX,
    GRID_NX, GRID_NY
)
from lighting_model import calculate_illuminance_matrix, get_grid_coordinates
from state import BulbState, RoomLightingState
from objective import evaluate_kpis, cost_function
from simulated_annealing import SimulatedAnnealingOptimizer


class TestOptimization(unittest.TestCase):
    """Pruebas unitarias automatizadas para el sistema de optimización con Temple Simulado."""

    def test_room_geometry_and_catalog(self):
        """Verifica las dimensiones de 56 m² y el catálogo de focos."""
        self.assertEqual(ROOM_WIDTH * ROOM_LENGTH, 56.0)
        self.assertEqual(ROOM_AREA, 56.0)
        self.assertEqual(NUM_BULBS, 6)
        self.assertGreaterEqual(len(BULB_CATALOG), 5)
        self.assertEqual(BULB_CATALOG[0].wattage, 0.0)  # Apagado

    def test_lighting_model_physics(self):
        """Valida los cálculos fotométricos y la ley inversa del cuadrado."""
        x, y, X, Y = get_grid_coordinates()
        self.assertEqual(X.shape, (GRID_NY, GRID_NX))
        
        # 1 foco encendido en el centro (3.5, 4.0) de 30W
        positions = np.array([[3.5, 4.0]] + [[0.5, 0.5]] * 5)
        models = np.array([5, 0, 0, 0, 0, 0]) # Solo foco 1 encendido (30W)
        
        E_mat = calculate_illuminance_matrix(positions, models)
        self.assertEqual(E_mat.shape, (GRID_NY, GRID_NX))
        self.assertTrue(np.all(E_mat >= 0.0))
        
        # El punto más brillante debe estar cerca del centro
        max_idx = np.unravel_index(np.argmax(E_mat), E_mat.shape)
        self.assertTrue(np.isclose(x[max_idx[1]], 3.5, atol=0.6))
        self.assertTrue(np.isclose(y[max_idx[0]], 4.0, atol=0.6))

    def test_state_and_vector_conversion(self):
        """Valida la serialización y deserialización vectorial del estado."""
        rng = np.random.default_rng(123)
        state = RoomLightingState.create_random(rng=rng)
        self.assertEqual(len(state.bulbs), 6)
        
        vector = state.to_vector()
        self.assertEqual(len(vector), 6 * 3)  # 18 valores
        
        # Reconstrucción
        decoded_state = RoomLightingState.from_vector(vector)
        self.assertEqual(len(decoded_state.bulbs), 6)
        
        for b_orig, b_dec in zip(state.bulbs, decoded_state.bulbs):
            self.assertTrue(np.isclose(b_orig.x, b_dec.x, atol=1e-5))
            self.assertTrue(np.isclose(b_orig.y, b_dec.y, atol=1e-5))
            self.assertEqual(b_orig.model_idx, b_dec.model_idx)

    def test_neighbor_operator(self):
        """Verifica que el operador de vecindario produzca estados válidos dentro de los límites."""
        rng = np.random.default_rng(456)
        parent = RoomLightingState.create_random(rng=rng)
        
        neighbor = parent.get_neighbor(rng=rng)
        self.assertEqual(len(neighbor.bulbs), 6)
        for b in neighbor.bulbs:
            self.assertTrue(0.0 <= b.x <= ROOM_WIDTH)
            self.assertTrue(0.0 <= b.y <= ROOM_LENGTH)
            self.assertTrue(0 <= b.model_idx < len(BULB_CATALOG))

    def test_kpis_and_cost_functions(self):
        """Verifica el cálculo coherente de los KPIs y la función de costo heurístico."""
        rng = np.random.default_rng(789)
        state = RoomLightingState.create_random(rng=rng)
        kpis = evaluate_kpis(state)
        
        self.assertGreaterEqual(kpis.avg_lux, 0.0)
        self.assertTrue(0.0 <= kpis.uniformity <= 1.0)
        self.assertTrue(0.0 <= kpis.coverage_pct <= 100.0)
        self.assertGreaterEqual(kpis.total_watts, 0.0)
        self.assertTrue(0 <= kpis.active_bulbs <= 6)
        self.assertGreaterEqual(kpis.heuristic_cost, 0.0)
        
        # Comprobar consistencia con función helper
        self.assertTrue(np.isclose(cost_function(state), kpis.heuristic_cost))

    def test_simulated_annealing_optimizer_runs(self):
        """Valida la ejecución y convergencia del Temple Simulado."""
        optimizer = SimulatedAnnealingOptimizer(
            initial_temp=1.5,
            final_temp=0.01,
            cooling_rate=0.8,
            steps_per_temp=5,
            seed=42
        )
        result = optimizer.optimize()
        
        self.assertIsNotNone(result.best_state)
        self.assertLessEqual(result.best_kpis.heuristic_cost, result.initial_kpis.heuristic_cost + 1e-4)
        self.assertGreater(len(result.history), 0)
        self.assertGreater(result.execution_time_sec, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


