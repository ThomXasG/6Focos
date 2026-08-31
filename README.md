# Optimización de Iluminación y Costo: 6 Focos en Habitación de 56 m²

**Materia:** Inteligencia Artificial - Séptimo Semestre  
**Tema:** Búsquedas Locales (Clase 4)  
**Algoritmo Implementado:** Temple Simulado (*Simulated Annealing*)  

---

## 1. Descripción del Problema
En una habitación de **$56\text{ m}^2$** ($7.0\text{ m} \times 8.0\text{ m}$, altura de techo $2.8\text{ m}$), se encuentran distribuidos **6 focos** de forma aleatoria.

El objetivo es:
1. **Maximizar la iluminación:** Lograr un nivel de iluminancia óptimo y confortable ($\sim 300\text{ Lux}$ en el plano de trabajo según la norma UNE-EN 12464-1), maximizando la uniformidad ($U_0 = E_{min}/E_{avg}$) y evitando zonas oscuras o sobreiluminadas.
2. **Minimizar el costo:** Reducir la potencia eléctrica consumida (Watts) y el costo mensual de energía, apagando focos redundantes o ajustando su vatiaje comercial (LED de 0W, 9W, 12W, 15W, 20W, 30W).

---

## 2. Fundamento Teórico (Clase 4 - Búsquedas Locales)

### ¿Por qué Temple Simulado (*Simulated Annealing*)?
Tal como se explicó en la **Clase 4**:
- El **Ascenso de Colinas clásico** es voraz y se estanca en óptimos locales aproximadamente el **86% de las veces**.
- El **Temple Simulado** se inspira en el recocido térmico de metales/vidrio. Utiliza un parámetro de **Temperatura ($T$)** y permite aceptar estados peores probabilísticamente mediante la distribución de Boltzmann:
  $$P = e^{-\frac{\Delta h(s)}{T}}$$
- Si un nuevo estado vecino mejora el costo ($\Delta h < 0$), se acepta de inmediato.
- Si un nuevo estado vecino empeora el costo ($\Delta h \ge 0$), se calcula la probabilidad $P$. Si un número aleatorio uniforme $r \in [0, 1)$ es menor que $P$, el estado se acepta.
- La temperatura disminuye gradualmente en cada época mediante un factor de enfriamiento geométrico:
  $$T_{k+1} = \alpha \cdot T_k \quad (\alpha = 0.95)$$
- Al inicio (alta temperatura $T_0 = 1.5$), el algoritmo explora ampliamente el espacio de estados aceptando soluciones peores para escapar de mínimos locales. Al final (baja temperatura $T_{min} = 10^{-4}$), se concentra en la explotación pura (afinación de la solución óptima).

---

## 3. Modelo Matemático y Fotométrico

### Ley Inversa del Cuadrado y Ley del Coseno de Lambert
Para cada punto $(x, y)$ sobre el plano de trabajo (a $0.75\text{ m}$ del suelo, altura efectiva $h = 2.05\text{ m}$), la iluminancia debida al foco $i$ en posición $(x_i, y_i)$ con intensidad luminosa $I_i$ es:
$$d_i = \sqrt{(x - x_i)^2 + (y - y_i)^2 + h^2}$$
$$E_i(x,y) = \frac{I_i \cdot \cos\theta_i}{d_i^2} = \frac{I_i \cdot h}{d_i^3} \quad (\text{en Lux})$$
$$E_{total}(x,y) = \sum_{i=1}^{6} E_i(x,y)$$

### Función Heurística de Costo $h(s)$ (A Minimizar)
$$h(s) = w_1 \cdot \text{Penalización\_Déficit\_Lux} + w_2 \cdot \text{Penalización\_Falta\_Uniformidad} + w_3 \cdot \text{Penalización\_Potencia}$$

---

## 4. Estructura Modular del Proyecto

El proyecto está diseñado de forma desacoplada y orientada a objetos:
1. **Configuración (`config.py`):** Constantes físicas, catálogo de focos LED, pesos de la heurística y parámetros del algoritmo.
2. **Modelo Fotométrico (`lighting_model.py`):** Motor de cálculo físico vectorizado con NumPy sobre una malla de 224 puntos.
3. **Espacio de Estados (`state.py`):** Representación del estado (`BulbState`, `RoomLightingState`) y generación de vecinos estocásticos.
4. **Función Objetivo y KPIs (`objective.py`):** Evaluación de métricas de calidad lumínica y cálculo de la función heurística $h(s)$.
5. **Algoritmo de Temple Simulado (`simulated_annealing.py`):** Motor de optimización estocástica con criterio de Metrópolis.
6. **Visualización (`visualizer.py`):** Generador de mapas de calor 2D, gráficos de convergencia y barras comparativas.
7. **Pruebas Automatizadas (`tests/test_optimization.py`):** Suite de tests unitarios que validan física, geometría y convergencia.
8. **Programa Principal (`main.py`):** Orquestador de la ejecución completa y reporte de resultados.

---

## 5. Ejecución del Proyecto

### Ejecutar Pruebas Unitarias
```bash
python tests/test_optimization.py
# o también:
pytest tests/ -v
```

### Ejecutar la Optimización Principal
```bash
python main.py
```

### Gráficos Generados (`output/`):
- `output/mapa_iluminacion_comparativo.png`: Mapas de calor 2D con isolíneas de luxes y posiciones de los focos antes vs después.
- `output/curva_temple_simulado.png`: Curvas de descenso de temperatura y reducción del costo $h(s)$ por época.
- `output/comparacion_kpis.png`: Gráfico de barras comparando todos los KPIs.
