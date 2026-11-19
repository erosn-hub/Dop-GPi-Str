# neuroflex_stats — Programa Estadístico Modular para Neurociencia Conductual, Neural y Videográfica

Paquete modular en Python diseñado para evaluar tendencias longitudinales, correlaciones multivariadas, microestructura conductual, dinámica de aprendizaje y actividad neural/cinemática en experimentos de neurociencia de sistemas (optogenética, fotometría de fibra e imagenología de calcio enfocadas en el circuito **GPi/EPN → Estriado**).

## Estructura del Paquete

```text
neuroflex_project/
├── neuroflex_stats/
│   ├── __init__.py           # Exportaciones públicas del paquete
│   ├── core.py               # Metadatos, geometría arena, control de peso (~80% cada 48h) y registro extensible (@register_task)
│   ├── behavior_stats.py     # Microestructura de licking, curvas espacio-estado (Smith 2004), psicometría/cronometría y LMM
│   ├── neural_video.py       # PETH con ajuste isosbéstico 405nm + bootstrap/permutaciones, cinemática video y Regresión Ridge
│   └── viz_and_pipeline.py   # Gráficos de grado publicación (300 DPI) y orquestador run_complete_pipeline
├── run_example.py            # Script ejecutable completo con datos sintéticos de demostración
├── requirements.txt          # Dependencias estándar de Python
└── README.md                 # Guía de instalación, mapeo de datos reales y extensión
```

## Instalación y Ejecución Rápida

1. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecuta la demostración completa de extremo a extremo:
   ```bash
   python run_example.py
   ```
   Esto generará la carpeta `outputs_neuroflex/` con:
   - `experiment_metadata_audit.json` (parámetros, pesos corporales por sesión, geometría y notas bibliográficas).
   - `figures/fig1_learning_psychometrics.png` (dinámica de aprendizaje espacio-estado, curva psicométrica y cronométrica).
   - `figures/fig2a_photometry_peth_tone.png` y `figures/fig2b_photometry_peth_vertex.png` (PETH peri-evento con intervalos de confianza bootstrap y marcas de umbral consecutivo).
   - `figures/fig3_correlations_microstructure_ridge.png` (matriz de correlaciones Spearman, histograma de ILI y varianza única $\Delta R^2$).

## Uso con tus Datos Reales (CSV / HDF5 / NWB)

Carga tus tablas desde CSV (`pd.read_csv(...)`) respetando las columnas requeridas:

- **`trials_df`**: `trial_id`, `session_id`, `mouse_id`, `trial_start_s`, `vertex_enter_s`, `vertex_hold_ms`, `hold_target_ms`, `tone_onset_s`, `tone_freq_hz`, `first_lick_s`, `reward_delivered_ul`, `outcome` (`"hit"` o `"premature_exit"`). Opcionales: `opto_laser_on`, `session_num`, `group_condition`.
- **`licks_df`**: `session_id`, `mouse_id`, `trial_id`, `lick_timestamp_s`.
- **`photometry_df`**: `timestamp_s`, `signal_465` (o `dff` si ya está preprocesado en imagenología de calcio), y opcionalmente `isosbestic_405`.
- **`pose_df`** (opcional, desde DeepLabCut / SLEAP): `frame_idx`, `timestamp_s`, `nose_x`, `nose_y`, `midback_x`, `midback_y` (y columnas `*_likelihood`).

Luego invoca:

```python
from pathlib import Path
from neuroflex_stats import ExperimentMetadata, run_complete_pipeline

results = run_complete_pipeline(
    task_name="vertex_cue_lick_3input",
    metadata=metadata,
    trials_df=trials_df,
    licks_df=licks_df,
    photometry_df=photometry_df,
    pose_df=pose_df,
    output_dir=Path("resultados_reales"),
)
```

## Cómo Añadir Nuevas Pruebas Conductuales a Futuro

Sin modificar ningún archivo del núcleo, crea un nuevo script o módulo que herede de `BehaviorTaskProtocol` y decóralo con `@register_task("mi_nueva_tarea")`:

```python
from neuroflex_stats import BehaviorTaskProtocol, register_task

@register_task("secuencia_motora_2palancas")
class SecuenciaMotoraTask(BehaviorTaskProtocol):
    def validate_and_preprocess(self, trials_df):
        # Valida columnas y calcula is_hit, reaction_transit_ms, etc.
        return trials_df

    def alignment_events(self):
        return {"Inicio": "trial_start_s", "Recompensa": "reward_s"}

    def primary_stimulus_column(self):
        return "sequence_length"
```
