# Clasificación de enfermedades foliares del tomate en *edge* (Jetson Nano)

Material de **reproducibilidad** del Trabajo Fin de Máster sobre el despliegue local de un
clasificador compacto de enfermedades del tomate en computación periférica, con adaptación de
dominio progresiva, barrido de hiperparámetros (frente de Pareto), optimización numérica
(FP16 / INT8), poda estructural y medición física sobre hardware real.

Este repositorio contiene **material de trazabilidad y apoyo a la reproducción del experimento**:
código, configuraciones, predicciones, resultados de medición y modelos finales o candidatos.
La reproducción completa requiere descargar los datasets desde sus fuentes y regenerar los
motores TensorRT en el dispositivo objetivo.

---

## 1. Pipeline experimental

```
Zenodo (laboratorio, 10 clases)  ──►  PlantDoc (campo)  ──►  Umair-Pirzada (dominio objetivo)
        dominio base               adaptación + validación          evaluación inicial + hard cases
```

- **Hilo A — Adaptación de dominio progresiva:** `baseline` → `fieldaug` → `domain_mix_pd5`
  → `pd5 + hard cases Umair`, sobre `YOLOv8n-cls`. Optimización numérica (FP16/INT8) y poda
  estructural ligera con recuperación.
- **Hilo B — Barrido Pareto multiobjetivo:** 12 configuraciones (resolución, *lr*, *weight decay*,
  *dropout*, *scheduler* coseno, perfil de aumentación, MixUp/CutMix, *freeze*, optimizador),
  evaluadas en los tres dominios.
- **Bloque transversal — Edge y medición física:** exportación a ONNX (opset 12), compilación
  TensorRT (FP32/FP16/INT8), instrumentación con `tegrastats` y sensores INA3221, y benchmark
  repetido e intercalado. El orden base → poda → recuperación se mantiene fijo y no constituye
  un contrabalanceo completo.

### Alcance metodológico

- `PlantDoc val` contiene 69 imágenes y se utiliza durante la validación y selección de
  configuraciones. Sus métricas son estimaciones exploratorias, no un test final independiente.
- `Umair valid` se evalúa inicialmente sobre `pd5`, pero después interviene en la minería de
  *hard cases*. La mejora posterior caracteriza adaptación al dominio objetivo, no generalización
  sobre datos completamente no vistos.
- En el Hilo B, 10 de las 12 configuraciones son no dominadas en los tres objetivos de exactitud.
  Las cinco variantes p02, p03, p05, p07 y p08 forman un subconjunto operativo llevado a Jetson;
  no constituyen por sí solas todo el frente de Pareto matemático.
- El barrido utiliza una semilla por configuración. Además, la exactitud post-TensorRT de p02,
  p03 y p07 está pendiente; por ello, la selección de modelo es provisional.
- `trtexec` mide el motor TensorRT con entrada sintética y el motor cargado. No incluye cámara,
  decodificación, preprocesado, postprocesado ni interfaz. INA3221 estima potencia de placa y no
  sustituye una medición externa calibrada del sistema completo.

## 2. Estructura del repositorio

```
codigo/
  scripts/                          Scripts del experimento (entrenamiento, preparación de datos,
                                    exportación, cuantización INT8, poda, evaluación, benchmark, análisis)
modelos/                            Modelos finales y candidatos curados (.pt y .onnx opset 12)
  domainmix_pd5/                    modelo domain_mix_pd5
  domainmix_pd5_umair_hard_e25/     modelo base del Hilo A (pd5 + hard cases)
  e25/                              baseline/fieldaug e25
  pareto_candidates/                candidatos edge p02, p03, p05, p07, p08
resultados/
  clasificacion/                    Métricas por época (results.csv) y configuración (args.yaml) de cada run
  evaluacion_externa/               Validación por dominios y análisis de errores (nombre histórico)
  pareto/                           Plan del barrido, resultados y análisis del frente de Pareto
  poda/                             Resultados de poda y recuperación
  jetson/                           Mediciones en Jetson Nano: trtexec, tegrastats e INA3221 (CSV/log/JSON)
requirements.txt
```

## 3. Conjuntos de datos (no incluidos — descargar de la fuente)

| Conjunto | Uso | Origen |
|---|---|---|
| Zenodo (record **8311631**) | Entrenamiento + validación base | https://doi.org/10.5281/zenodo.8311631 |
| PlantDoc (subconjunto tomate) | Adaptación + validación de dominio | https://github.com/pratikkayal/PlantDoc-Dataset |
| Umair-Pirzada | Evaluación inicial + minería de *hard cases* | Repositorio público enlazado desde GitHub/Kaggle |

La construcción de los conjuntos compuestos (`pd5`, *hard cases*, subconjuntos de calibración
INT8) es **reproducible** con los scripts `codigo/scripts/prepare_*.py`.

## 4. Entorno

- **PC (entrenamiento y exportación):** Windows · Python 3.12.10 · PyTorch 2.12.0+cu126 ·
  Ultralytics 8.4.50 · ONNX 1.21.0 (opset 12) · NVIDIA RTX 3080. Ver `requirements.txt`.
- **Edge (inferencia y medición):** Advantech MIC-710AI (Jetson Nano, Tegra X1) · L4T R32.6.1 ·
  JetPack 4.6 · CUDA 10.2 · cuDNN 8.2.1 · TensorRT 8.0.1.6 · modo MAXN.

> La compatibilidad ONNX **opset 12** ↔ **TensorRT 8.0** es obligatoria: opsets más recientes
> generan operadores no soportados en JetPack 4.6.

## 5. Reproducción (resumen)

```text
# 1. Entorno (PC Windows)
python -m venv .venv
powershell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Preparar datasets compuestos (tras descargar las fuentes)
python codigo/scripts/prepare_plantdoc_tomato.py
python codigo/scripts/prepare_umair_external.py
python codigo/scripts/prepare_domain_mix_dataset.py
python codigo/scripts/prepare_domain_mix_hardcases.py

# 3. Hilo A (entrenamiento y adaptación de dominio)
python codigo/scripts/train_cls.py             # baseline / fieldaug / domain_mix / pd5+hard
python codigo/scripts/eval_cls_external.py      # exactitud por dominio

# 4. Hilo B (barrido Pareto) y poda
python codigo/scripts/run_pareto_sweep.py
python codigo/scripts/prune_head_finetune_cls.py
python codigo/scripts/analyze_pareto_results.py

# 5. Exportación y cuantización
python codigo/scripts/export_cls.py
python codigo/scripts/build_trt_int8_engine.py    # (en el Jetson)

# 6a. Orquestación desde el PC Windows (requiere PuTTY/plink/pscp)
powershell -ExecutionPolicy Bypass -File .\codigo\scripts\run_jetson_trt_benchmark.ps1 `
  -JetsonHost <host> -JetsonUser <usuario> -JetsonPassword <contraseña> `
  -OnnxPath <modelo.onnx> -PtPath <modelo.pt> -LocalResultsDir <directorio-resultados>

# 6b. Medición ejecutada en el dispositivo Jetson
bash codigo/scripts/run_resource_benchmark_jetson.sh
bash codigo/scripts/run_power_benchmark_jetson.sh

# 6c. Posprocesado (sustituir los marcadores por los artefactos generados)
python codigo/scripts/parse_tegrastats.py <tegrastats.log> [...] --out <resource_summary.csv>
python codigo/scripts/parse_power_samples.py <ina3221.csv> [...] --out <power_summary.csv>
python codigo/scripts/summarize_alternating_benchmark.py `
  --trtexec <trtexec_summary.csv> --resources <resource_summary.csv> `
  --power <power_summary.csv> --out-csv <combined.csv> --out-md <summary.md>
```

> **Seguridad:** los parámetros de conexión deben proporcionarse en cada entorno. No deben
> versionarse contraseñas reales ni reutilizarse credenciales expuestas en historiales previos.

## 6. Qué **no** está aquí y por qué

- **Datasets** (`dataset/`, `downloads/` ~ 5 GB): se descargan de las fuentes (sección 3).
- **`.venv`**: entorno virtual, regenerable con `requirements.txt`.
- **Checkpoints intermedios** (`runs/**/weights/*.pt`, 66 ficheros): redundantes; los modelos
  finales relevantes están en `modelos/`.
- **Motores TensorRT** (`.engine`): se construyen en el propio Jetson y son específicos del
  hardware/driver; se regeneran con `build_trt_int8_engine.py` y `export_cls.py`.

Los README de resultados conservan algunas rutas absolutas `F:\TFM experimento\...` y
`/home/mic-710ai/...` como procedencia de la ejecución original. No son rutas requeridas para
clonar el repositorio; los scripts públicos equivalentes están en `codigo/scripts/` y los
artefactos curados en `modelos/` y `resultados/`.


## 7. Correspondencia con el Anexo A (Trazabilidad de artefactos) de la memoria

| Anexo A | En este repositorio |
|---|---|
| `training/runs/classify/runs/` | `resultados/clasificacion/` (métricas; pesos finales en `modelos/`) |
| `training/runs/pareto_sweep/` | `resultados/pareto/` |
| `training/runs/pruning/` | `resultados/poda/` |
| `training/exports/` (ONNX) | `modelos/` |
| `jetson-*-results/`, `jetson_pareto_candidates/` | `resultados/jetson/` |
| `training/scripts/` | `codigo/scripts/` |
| Motores TensorRT en el Jetson | No incluidos (regenerables en el dispositivo) |
