# Experimento de poda ligera de cabeza - YOLOv8n-cls tomate

> **Alcance.** Esta variante pertenece al Hilo A. PlantDoc val se usa durante el desarrollo y
> contiene 69 imágenes. El rendimiento se mide con `trtexec` sobre el motor cargado y no incluye
> el pipeline completo. La comparación global con p02 y p03 sigue siendo provisional porque falta
> su exactitud post-TensorRT.

## Objetivo

Evaluar una primera poda estructurada prudente sobre el modelo `domain mix pd5 + Umair hard cases`, reduciendo canales de la cabeza de clasificación y midiendo el impacto en precisión, tamaño, rendimiento y consumo en Jetson Nano.

## Variante generada

- Modelo base: `tomato_yolov8n_cls_domainmix_pd5_umair_hard_e25.pt`.
- Script: `F:\TFM experimento\training\scripts\prune_head_finetune_cls.py`.
- Modelo podado local: `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_e5\head_pruned_init.pt`.
- ONNX exportado: `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_e5\head_pruned_init.onnx`.
- Canales de la cabeza: 1280 -> 1024.
- Parámetros: 1.451.098 -> 1.382.490.
- Reducción de parámetros: 68.608, equivalente a 4,73 %.

## Precisión local previa a TensorRT

| Modelo | Dataset | Top-1 |
|---|---|---:|
| Base `pd5 + hard cases` | PlantDoc val | 0,623 |
| Head-prune 20% PT | PlantDoc val | 0,623 |
| Head-prune 20% PT | Umair valid mapeado | 0,940 |

La poda inicial conserva la precisión local en PlantDoc. La recuperación corta de 5 épocas no se considera resultado principal porque Ultralytics reconstruyó el entrenamiento desde la ruta original y produjo una variante degradada; se conserva como evidencia exploratoria, no como modelo candidato.

## Despliegue TensorRT en Jetson Nano

Archivos remotos:

- ONNX: `/home/mic-710ai/tfm-experimento/models/tomato_yolov8n_cls_domainmix_pd5_umair_hard_headprune20_opset12.onnx`.
- Engine FP32: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_headprune20_fp32.engine`.
- Engine FP16: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_headprune20_fp16.engine`.

Build TensorRT:

| Precisión | Tamaño del motor | Tiempo build |
|---|---:|---:|
| FP32 | 8 MB | 96,698 s |
| FP16 | 5 MB | 221,279 s |

## Validación funcional TensorRT

| Variante | Dataset | Correctas / total | Top-1 |
|---|---|---:|---:|
| Head-prune 20% TensorRT FP32 | PlantDoc val | 40 / 69 | 0,580 |
| Head-prune 20% TensorRT FP16 | PlantDoc val | 40 / 69 | 0,580 |

La precisión del motor TensorRT queda por debajo de la evaluación local del `.pt` podado. Esta diferencia debe tratarse como resultado experimental relevante.

## Rendimiento y consumo FP16

Medición con `trtexec --loadEngine`, `tegrastats` y muestreo INA3221:

| Métrica | Valor |
|---|---:|
| Throughput | 277,101 qps |
| Latencia media | 3,597 ms |
| P99 latencia | 3,806 ms |
| GPU compute medio | 3,534 ms |
| RAM media | 3049,5 MB |
| GPU media | 75,3 % |
| CPU media | 31,9 % |
| POM_5V_IN media | 6,35 W |
| Energía estimada de entrada | 146,2 J |

## Lectura

La poda ligera de cabeza mejora claramente el rendimiento respecto al FP16 base previo, pero introduce una pérdida de precisión al pasar por TensorRT sobre PlantDoc. Dentro del Hilo A representa la opción orientada a máximo throughput. En la comparación global del TFM no existe un único candidato final: p02 ofrece el equilibrio provisional, p03 prioriza la exactitud de validación y la variante podada prioriza rendimiento, a falta de completar la exactitud post-TensorRT de los candidatos del Hilo B.

## Evidencia

- Logs Jetson: `F:\TFM experimento\jetson-head-prune20-results\head_prune20`.
- Metadatos de poda: `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_e5\pruning_metadata.json`.
- Evaluaciones locales: `F:\TFM experimento\training\runs\external_eval\head_prune20_init_plantdoc_val` y `F:\TFM experimento\training\runs\external_eval\head_prune20_init_umair_valid`.
