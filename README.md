# qcar2_planner (ROS 2 Humble)

Paquete para:

1. Fusionar `/map` + malla de nvblox en un `GridMap` con incertidumbre.
2. Detectar **puntos de interés** en fronteras/intersecciones para explorar caminos no descubiertos.
3. Publicar métricas estandarizadas de progreso de mapeo.
4. Habilitar el planificador direccional cuando el mapa esté suficientemente cubierto.

> Guardrail: este paquete **NO** controla motores.

---

## Nodos

- `map_processor_node.py`
	- Publica:
		- `/grid_map`
		- `/planner_occupancy`
		- `/planner_uncertainty`
	- Suscribe:
		- `/nvblox_node/mesh_marker` (`visualization_msgs/MarkerArray`)
- `exploration_manager_node.py`
	- Publica:
		- `/exploration_goal` (`PoseStamped`)
		- `/frontier_markers` (`MarkerArray`)
		- `/exploration_metrics` (`Float32MultiArray`)
- `directional_planner_server.py`
	- Servicios:
		- `/enable_planner` (`SetBool`)
		- `/get_directional_path` (`GetDirectionalPath`)

---

## Métricas estandarizadas

`/exploration_metrics` usa un vector fijo (`exploration_metrics_v2`):

- `[0]` `mapped_pct`
- `[1]` `explored_pct`
- `[2]` `mean_uncertainty`
- `[3]` `free_cells`
- `[4]` `interest_points`
- `[5]` `reachable_interest_points`
- `[6]` `selected_goal_dist_m`
- `[7]` `robot_x`
- `[8]` `robot_y`
- `[9]` `state` (`0=MAPPING`, `1=READY`)

---

## Mejoras implementadas

- Detección de frontera tipo Nav2: **celda desconocida con vecino libre**.
- Priorización en tiempo real de frontera **más cercana y alcanzable**.
- Filtrado por **intersecciones** usando sectores angulares desconocidos.
- Selección de objetivo con histéresis (distancia/tiempo + margen de cambio) para evitar oscilaciones.
- Caché de máscara de área de trabajo para reducir cómputo repetido.
- Actualización incremental de incertidumbre en `map_processor_node` (solo celdas tocadas).
- Reducción de frecuencia de publicación por defecto para bajar carga de CPU.

