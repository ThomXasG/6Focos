# Documentación Técnica: Explicación de Cada Archivo del Proyecto

**Proyecto:** Optimización de Iluminación y Costo (6 Focos en Habitación de 56 m²)  
**Materia:** Inteligencia Artificial - Séptimo Semestre  
**Algoritmo:** Búsqueda Local mediante Temple Simulado (*Simulated Annealing* - Clase 4)  

---

## 1. Diagrama de Arquitectura y Flujo del Sistema

El proyecto sigue una arquitectura modular y desacoplada, donde cada archivo cumple una responsabilidad única:

```mermaid
flowchart TD
    CONFIG[config.py<br/><i>Parámetros, Geometría, Catálogo y Pesos</i>] --> MODEL[lighting_model.py<br/><i>Física Fotométrica de Lambert</i>]
    CONFIG --> STATE[state.py<br/><i>Espacio de Estados y Vecindarios</i>]
    CONFIG --> OBJ[objective.py<br/><i>KPIs y Función Heurística h(s)</i>]
    CONFIG --> SA[simulated_annealing.py<br/><i>Temple Simulado</i>]
    
    STATE --> OBJ
    MODEL --> OBJ
    
    STATE --> SA
    OBJ --> SA
    
    SA --> MAIN[main.py<br/><i>Punto de Entrada y Orquestador</i>]
    STATE --> MAIN
    OBJ --> MAIN
    
    MAIN --> VIS[visualizer.py<br/><i>Mapas de Calor 2D y Gráficos</i>]
    VIS --> OUT[(Carpeta output/<br/><i>Gráficos PNG</i>)]
    
    TEST[tests/test_optimization.py<br/><i>Pruebas Unitarias</i>] -.-> STATE
    TEST -.-> MODEL
    TEST -.-> OBJ
    TEST -.-> SA
```

---

## 2. Explicación Detallada de Cada Archivo

```
6Focos/
├── config.py                  # Constantes, catálogo LED y parámetros térmicos
├── lighting_model.py          # Motor físico de cálculo fotométrico (Lambert)
├── state.py                   # Modelado del espacio de estados y vecindarios
├── objective.py               # KPIs fotométricos, económicos y función de costo h(s)
├── simulated_annealing.py     # Algoritmo de Búsqueda Local (Temple Simulado)
├── visualizer.py              # Generación de mapas de calor 2D y gráficas
├── main.py                    # Script principal de ejecución y reporte
├── tests/
│   └── test_optimization.py  # Suite de pruebas unitarias automatizadas
├── README.md                  # Resumen general del proyecto
├── EXPLICACION_ARCHIVOS.md    # Este documento explicativo
└── output/                    # Gráficos generados de alta resolución
```

---

### 1. `config.py` — Configuración Global y Catálogo

**Propósito:** Centralizar todas las constantes físicas, geométricas, comerciales, económicas y los hiperparámetros del algoritmo en un único punto de control.

#### Contenido y Variables Clave:
1. **Geometría de la Habitación:**
   - `ROOM_WIDTH = 7.0` m (ancho en eje X) y `ROOM_LENGTH = 8.0` m (largo en eje Y) $\rightarrow$ Área total $= 56\text{ m}^2$.
   - `ROOM_HEIGHT = 2.8` m (altura del techo) y `WORKPLANE_HEIGHT = 0.75` m (altura de mesas de trabajo).
   - `EFFECTIVE_HEIGHT = 2.05` m (distancia vertical foco-plano de trabajo).
   - `MARGIN_WALL = 0.5` m (separación mínima para no adosar focos a esquinas/paredes).
   - `GRID_NX = 14`, `GRID_NY = 16` $\rightarrow$ Malla de $224$ puntos de muestreo distribuidos cada $0.5\text{ m}$.
2. **Catálogo Comercial de Focos LED (`BULB_CATALOG`):**
   - Modela modelos comerciales con su potencia en Watts ($W$), flujo luminoso en lúmenes ($lm$) e intensidad luminosa central en Candelas ($I$):
     - `Apagado (0W)`: $0\text{ W}$, $0\text{ lm}$, $0\text{ cd}$.
     - `LED 9W`: $9\text{ W}$, $800\text{ lm}$, $254.6\text{ cd}$.
     - `LED 12W`: $12\text{ W}$, $1100\text{ lm}$, $350.1\text{ cd}$.
     - `LED 15W`: $15\text{ W}$, $1500\text{ lm}$, $477.5\text{ cd}$.
     - `LED 20W`: $20\text{ W}$, $2000\text{ lm}$, $636.6\text{ cd}$.
     - `LED 30W`: $30\text{ W}$, $3000\text{ lm}$, $954.9\text{ cd}$.
3. **Normas de Iluminación y Costos:**
   - `TARGET_LUX = 300.0` Lux (norma UNE-EN 12464-1 para salas de estudio y trabajo).
   - `MIN_ACCEPTABLE_LUX = 250.0` Lux.
   - `TARGET_UNIFORMITY = 0.60` ($E_{min}/E_{avg}$).
   - `ELECTRICITY_RATE_KWH = 0.12` USD/kWh, $8\text{ h/día}$, $30\text{ días/mes}$.
4. **Parámetros del Temple Simulado (Clase 4):**
   - `SA_INITIAL_TEMP = 1.5` ($T_0$, calibrada para aceptar $\sim 80\text{-}90\%$ de movimientos iniciales).
   - `SA_FINAL_TEMP = 0.0001` ($T_{min}$, asegura congelamiento y convergencia fina).
   - `SA_COOLING_RATE = 0.95` ($\alpha$, factor de enfriamiento geométrico $T_{k+1} = \alpha T_k$).
   - `SA_STEPS_PER_TEMP = 35` (pasos de búsqueda por nivel térmico).
   - `SA_PERTURBATION_STEP = 0.60` m (paso máximo de desplazamiento de focos).
5. **Ponderaciones de la Función Heurística $h(s)$:**
   - `WEIGHT_LUX_DEFICIT = 0.45` ($45\%$ para alcanzar los 300 Lux).
   - `WEIGHT_UNIFORMITY = 0.30` ($30\%$ para evitar zonas oscuras).
   - `WEIGHT_POWER_COST = 0.25` ($25\%$ para ahorro energético).

---

### 2. `lighting_model.py` — Motor Físico Fotométrico

**Propósito:** Implementar las leyes físicas de la óptica geométrica para calcular con exactitud la iluminancia en Luxes recibida en cada punto de la habitación.

#### Fundamento Físico:
Aplica la **Ley Inversa del Cuadrado** combinada con la **Ley del Coseno de Lambert**:
Para un punto $(x, y)$ en el plano de trabajo y un foco $i$ en $(x_i, y_i)$ con intensidad $I_i$:
$$d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2 + h^2}$$
$$\cos\theta_i = \frac{h}{d_i}$$
$$E_i(x, y) = \frac{I_i \cdot \cos\theta_i}{d_i^2} = \frac{I_i \cdot h}{d_i^3} \quad (\text{en Lux})$$
$$E_{total}(x, y) = \sum_{i=1}^{6} E_i(x, y)$$

#### Funciones:
- `get_grid_coordinates()`: Genera las matrices 2D (`X`, `Y`) de la malla de muestreo de $7\text{m} \times 8\text{m}$.
- `calculate_illuminance_matrix(bulbs_positions, bulb_catalog_indices)`: Calcula vectorizadamente con NumPy la matriz total de iluminancia de dimensión $(16, 14)$.

---

### 3. `state.py` — Espacio de Estados y Vecindarios

**Propósito:** Definir la estructura de datos que representa una solución (la configuración de los 6 focos) y los operadores para generar estados vecinos en la búsqueda local.

#### Clases y Métodos:
- **`BulbState` (Dataclass):**
  - Atributos: `x`, `y` (coordenadas en metros) y `model_idx` (índice del foco en el catálogo).
  - `clamp_bounds()`: Limita las coordenadas dentro de los márgenes $[0.5, 6.5]\text{ m}$ en X y $[0.5, 7.5]\text{ m}$ en Y.
- **`RoomLightingState`:**
  - Encapsula la lista de los 6 focos (`self.bulbs`).
  - `create_random()`: Genera un estado inicial aleatorio dentro de la habitación de $56\text{ m}^2$.
  - `get_positions_array()`: Retorna matriz NumPy de $(6, 2)$ con las posiciones.
  - `get_model_indices_array()`: Retorna vector NumPy de 6 enteros con los modelos.
  - `get_total_wattage()`: Suma la potencia activa en Watts.
  - `get_active_bulbs_count()`: Cuenta focos encendidos ($>0\text{W}$).
  - `to_vector()` y `from_vector()`: Serializa/deserializa el estado a un vector 1D de 18 elementos $[x_1, y_1, \text{idx}_1, \dots, x_6, y_6, \text{idx}_6]$.
  - `get_neighbor(perturbation_step)`: **Generador de vecindario estocástico.** Selecciona aleatoriamente 1 o 2 focos y aplica:
    1. Desplazamiento espacial en $(x, y)$ ($60\%$ de probabilidad).
    2. Cambio de potencia en $\pm 1$ nivel de catálogo ($25\%$ de probabilidad).
    3. Encendido/Apagado directo ($15\%$ de probabilidad).

---

### 4. `objective.py` — Función Objetivo y Factores Medibles (KPIs)

**Propósito:** Evaluar cuantitativamente la calidad de una configuración mediante indicadores clave de rendimiento (KPIs) fotométricos y económicos, consolidándolos en la función de costo heurístico $h(s)$.

#### Clases y Funciones:
- **`LightingKPIs` (Dataclass):**
  - Contiene: `avg_lux`, `min_lux`, `max_lux`, `uniformity` ($U_0 = E_{min}/E_{avg}$), `coverage_pct` ($\%\ge 270\text{ lx}$), `total_watts`, `active_bulbs`, `monthly_cost_usd` y `heuristic_cost`.
- **`evaluate_kpis(state)`:**
  1. Calcula la matriz de iluminancia $E(x, y)$.
  2. Calcula métricas estadísticas (media, mínimo, máximo, uniformidad, cobertura).
  3. Calcula el consumo mensual en dólares:
     $$\text{Costo Mensual} = \left(\frac{\text{Watts}}{1000}\right) \times 8\text{ h/día} \times 30\text{ días} \times 0.12\text{ USD/kWh}$$
  4. Calcula las tres componentes de penalización de la función heurística:
     - **Penalización por Lux:** Cuadrática si falta luz $\left(\frac{300 - E_{avg}}{300}\right)^2$.
     - **Penalización por Falta de Uniformidad:** $(1 - U_0) + \text{proporción de puntos oscuros}$.
     - **Penalización por Potencia:** $\frac{\text{Watts}}{180\text{ W}}$.
  5. Consolida el costo heurístico $h(s)$ (a minimizar):
     $$h(s) = 0.45 \cdot \text{Penalización\_Lux} + 0.30 \cdot \text{Penalización\_Uniformidad} + 0.25 \cdot \text{Penalización\_Potencia}$$
- **`cost_function(state)`:** Función helper que retorna directamente $h(s)$.

---

### 5. `simulated_annealing.py` — Motor de Temple Simulado

**Propósito:** Implementar el algoritmo de Búsqueda Local estocástica basado en los fundamentos de la **Clase 4**, capaz de escapar de mínimos locales y converger a soluciones de alta calidad.

#### Fundamento Teórico del Algoritmo:
1. **Inicio:** Se parte de un estado inicial aleatorio $s$ con costo $h(s)$ y temperatura alta $T_0$.
2. **Generación de Vecino:** En cada paso se genera un vecino $s' \in Vecindario(s)$ con costo $h(s')$.
3. **Variación de Costo:** $\Delta h = h(s') - h(s)$.
4. **Criterio de Metrópolis:**
   - Si $\Delta h < 0$ (el vecino mejora): Se acepta **incondicionalmente** ($s \leftarrow s'$).
   - Si $\Delta h \ge 0$ (el vecino empeora): Se calcula la probabilidad de Boltzmann:
     $$P = e^{-\frac{\Delta h}{T}}$$
     Se genera un número uniforme $r \in [0, 1)$. Si $r < P$, se acepta el movimiento peor para escapar de óptimos locales.
5. **Programa de Enfriamiento:** Al terminar cada época interna ($35$ pasos), la temperatura se reduce geométricamente:
   $$T_{k+1} = \alpha \cdot T_k \quad (\alpha = 0.95)$$
6. **Parada:** Cuando $T \le T_{min} = 10^{-4}$, el algoritmo se detiene y retorna el mejor estado histórico $s_{best}$.

#### Clases:
- `IterationLog`: Registro por época (temperatura, costo actual, mejor costo, tasa de aceptación).
- `SAResult`: Contenedor del resultado final (estado inicial, estado óptimo, KPIs comparativos, historial y tiempo de ejecución).
- `SimulatedAnnealingOptimizer`: Clase principal que ejecuta el método `optimize()`.

---

### 6. `visualizer.py` — Módulo de Visualización Gráfica

**Propósito:** Generar gráficos comparativos y mapas de calor 2D con calidad de publicación en tema oscuro (*dark mode*).

#### Funciones:
1. **`plot_comparative_heatmaps()` $\rightarrow$ `output/mapa_iluminacion_comparativo.png`:**
   - Renderiza 2 paneles lado a lado con mapas de contorno (`contourf`) e isolíneas (`contour`) de Luxes.
   - Dibuja la ubicación exacta $(x, y)$ de los 6 focos (diferenciando focos encendidos con su vatiaje vs focos apagados con 'X').
   - Añade un recuadro flotante con los KPIs fotométricos y económicos en cada panel.
2. **`plot_sa_convergence()` $\rightarrow$ `output/curva_temple_simulado.png`:**
   - **Panel Superior:** Curva de evolución de la función de costo $h(s)$ (costo actual vs mejor costo histórico) junto con la curva de temperatura en el eje secundario derecho.
   - **Panel Inferior:** Tasa de aceptación de Metrópolis ($\%$), mostrando la transición suave desde exploración activa ($100\%$) hasta explotación pura ($<5\%$).
3. **`plot_kpi_comparison_bars()` $\rightarrow$ `output/comparacion_kpis.png`:**
   - Gráfico de barras agrupadas comparando las métricas clave (Lux promedio, cobertura, uniformidad, potencia, costo mensual y costo heurístico).

---

### 7. `main.py` — Orquestador y Programa Principal

**Propósito:** Punto de entrada ejecutable del proyecto. Coordina la inicialización, la optimización, el reporte tabular en consola y la generación de gráficos.

#### Flujo de Ejecución:
1. Imprime el encabezado informativo del deber.
2. Genera el estado aleatorio inicial de 6 focos y evalúa sus KPIs.
3. Imprime la tabla de focos iniciales (posición, modelo LED, potencia y lúmenes).
4. Instancia `SimulatedAnnealingOptimizer` y ejecuta la búsqueda local con traza de progreso por épocas.
5. Imprime la tabla de focos optimizados.
6. Presenta la **Tabla Comparativa de KPIs** (Inicial vs Optimizado y Variación).
7. Invoca las 3 funciones de `visualizer.py` para guardar los gráficos en `output/`.

---

### 8. `tests/test_optimization.py` — Suite de Pruebas Unitarias

**Propósito:** Validar mediante pruebas automatizadas la corrección matemática, física y algorítmica del proyecto. Compatible con `pytest` y con el módulo nativo `unittest`.

#### Pruebas Implementadas:
| Prueba | Qué valida |
| :--- | :--- |
| `test_room_geometry_and_catalog` | Dimensiones exactas de $56\text{ m}^2$, 6 focos y catálogo comercial válido. |
| `test_lighting_model_physics` | Ley de Lambert, positividad de luxes y foco de máxima intensidad en el centro. |
| `test_state_and_vector_conversion` | Serialización bidireccional exacta Estado $\leftrightarrow$ Vector numérico 1D. |
| `test_neighbor_operator` | Que todas las perturbaciones mantengan los focos dentro de la habitación y con modelos válidos. |
| `test_kpis_and_cost_functions` | Coherencia numérica de los rangos de KPIs ($U_0 \in [0, 1]$, cobertura $\in [0, 100\%]$, $h(s) \ge 0$). |
| `test_simulated_annealing_optimizer_runs` | Ejecución completa del Temple Simulado y convergencia efectiva a un costo menor o igual. |

---

## 3. Guía de Ejecución

### Ejecución de Pruebas Unitarias
```bash
python tests/test_optimization.py
```

### Ejecución de la Optimización Principal
```bash
python main.py
```

### Archivos de Salida Generados
- `output/mapa_iluminacion_comparativo.png`
- `output/curva_temple_simulado.png`
- `output/comparacion_kpis.png`
