# Recuperación controlada tras poda ligera de cabeza

> **Alcance.** Este experimento pertenece al Hilo A. Las métricas de PlantDoc y Umair son
> validaciones de dominio dentro del desarrollo, no tests finales independientes. `trtexec` mide
> el motor TensorRT con entrada sintética y no una aplicación extremo a extremo.

## Objetivo

Comprobar si una recuperación corta tras la poda de cabeza del 20 % permite conservar la mejora de rendimiento observada en Jetson Nano sin perder precisión de validación.

## Entrenamiento de recuperación

- Script: `F:\TFM experimento\training\scripts\finetune_pruned_cls.py`.
- Modelo inicial: `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_e5\head_pruned_init.pt`.
- Salida: `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_torch_e5_lr1e4`.
- Épocas: 5.
- Learning rate: 0,0001.
- Batch: 64.
- Aumentación: activada.
- Mejor validación interna durante entrenamiento: 0,652 Top-1 en la época 3.

## Validación local del best.pt

| Dataset | Top-1 | Lectura |
|---|---:|---|
| Zenodo val | 0,986 | Prácticamente equivalente al modelo base. |
| PlantDoc val | 0,623 | No mejora frente al modelo base en el evaluador por dominios. |
| Umair valid mapeado | 0,945 | Ligera mejora frente al modelo base con hard cases. |

La mejora observada durante el entrenamiento no se reproduce como mejora neta en PlantDoc usando el evaluador basado en `YOLO.predict`.

## TensorRT FP16 en Jetson Nano

Archivos:

- ONNX: `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_torch_e5_lr1e4\best.onnx`.
- Engine remoto: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_headprune20_recovery_e5_fp16.engine`.

Build:

| Precisión | Tamaño del motor | Tiempo build |
|---|---:|---:|
| FP16 | 5 MB | 220,257 s |

Validación funcional TensorRT:

| Variante | Dataset | Correctas / total | Top-1 |
|---|---|---:|---:|
| Head-prune 20% recovery FP16 | PlantDoc val | 39 / 69 | 0,565 |

Rendimiento y consumo:

| Métrica | Valor |
|---|---:|
| Throughput | 277,185 qps |
| Latencia media | 3,595 ms |
| P99 latencia | 3,978 ms |
| GPU compute medio | 3,530 ms |
| RAM media | 2993,4 MB |
| GPU media | 76,7 % |
| CPU media | 32,3 % |
| POM_5V_IN media | 6,41 W |
| Energía estimada de entrada | 147,6 J |

## Lectura

La recuperación controlada conserva el perfil de rendimiento de la poda, pero no recupera la precisión del motor TensorRT. En esta iteración, la mejor lectura es metodológica: el flujo de poda + *fine-tuning* puede entrenarse y desplegarse, pero no mejora el compromiso precisión-rendimiento frente a la poda inicial ni frente al modelo FP16 sin poda.

Para el TFM, esta prueba sirve para delimitar que la poda ligera mejora latencia, pero que el despliegue TensorRT introduce una pérdida de precisión que no se corrige con una recuperación corta de 5 épocas.

## Evidencia

- Logs Jetson: `F:\TFM experimento\jetson-head-prune20-recovery-results\head_prune20_recovery`.
- Metadatos de entrenamiento: `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_torch_e5_lr1e4\finetune_metadata.json`.
- Validaciones por dominio:
  - `F:\TFM experimento\training\runs\external_eval\head_prune20_recovery_e5_plantdoc_val`.
  - `F:\TFM experimento\training\runs\external_eval\head_prune20_recovery_e5_umair_valid`.
  - `F:\TFM experimento\training\runs\external_eval\head_prune20_recovery_e5_zenodo_val`.
