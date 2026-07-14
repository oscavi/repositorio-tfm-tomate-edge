# Iteración de adaptación de dominio: Zenodo + PlantDoc ponderado

Fecha local: 2026-05-14.

> **Alcance.** Este README documenta una iteración del Hilo A. `PlantDoc val` se usa para
> validación durante el desarrollo y no constituye un test final independiente. Las métricas de
> `trtexec` caracterizan el motor TensorRT con entrada sintética; INA3221 estima potencia de placa.

## Objetivo

Comprobar si un ajuste adicional con imágenes PlantDoc, usadas como aproximación de imágenes de campo, mejora la robustez del clasificador de enfermedades de hojas de tomate sin degradar de forma relevante el rendimiento en el dominio original Zenodo.

## Dataset y entrenamiento

- Dataset base: `F:\TFM experimento\training\dataset\zenodo_tomato_leaf\tomato-dataset`.
- Dataset de campo: `F:\TFM experimento\training\dataset\plantdoc_tomato_cls`.
- Dataset mixto generado: `F:\TFM experimento\training\dataset\domain_mix_zenodo_plantdoc_pd5`.
- Estrategia: Zenodo train completo + PlantDoc train repetido 5 veces; validación con PlantDoc val.
- Punto de partida: `zenodo_tomato_cls_fieldaug_e25\weights\best.pt`.
- Entrenamiento de adaptación: 10 épocas con aumentación.
- Run local: `F:\TFM experimento\training\runs\classify\runs\domainmix_pd5_from_fieldaug_e25_e10`.

## Resultados de precisión

| Modelo | Zenodo val Top-1 | PlantDoc val Top-1 | Confianza media PlantDoc |
|---|---:|---:|---:|
| Field augmentation e25 | 0.985 | 0.290 | 0.889 |
| Domain mix pd5 desde e25 | 0.984 | 0.623 | 0.808 |

Interpretación: la adaptación de dominio mejora PlantDoc en 33,3 puntos porcentuales y mantiene Zenodo prácticamente estable (-0,1 puntos porcentuales).

## Exportación

- PyTorch: `F:\TFM experimento\training\exports\domainmix_pd5\tomato_yolov8n_cls_domainmix_pd5_from_e25.pt`.
- ONNX opset 12: `F:\TFM experimento\training\exports\domainmix_pd5\tomato_yolov8n_cls_domainmix_pd5_from_e25_opset12.onnx`.
- ONNX validado en PC:
  - Zenodo Top-1: 0.984.
  - PlantDoc Top-1: 0.623.

## Jetson Nano

- ONNX remoto: `/home/mic-710ai/tfm-experimento/models/tomato_yolov8n_cls_domainmix_pd5_from_e25_opset12.onnx`.
- PT remoto: `/home/mic-710ai/tfm-experimento/models/tomato_yolov8n_cls_domainmix_pd5_from_e25.pt`.
- Engine FP32: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_fp32.engine`.
- Engine FP16: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_fp16.engine`.
- Espacio libre tras la prueba: 1.7 GB en `/`.

## TensorRT

| Precisión | Tamaño | Build | Throughput build | Latencia build | P99 build | Throughput load | Latencia load | P99 load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 8.6 MB | 102.295 s | 196.891 qps | 5.066 ms | 5.381 ms | 195.418 qps | 5.105 ms | 5.449 ms |
| FP16 | 5.2 MB | 230.415 s | 259.084 qps | 3.848 ms | 4.175 ms | 261.452 qps | 3.814 ms | 3.901 ms |

FP16 frente a FP32 en la medición con el motor cargado:

- Latencia media: -25.3%.
- Throughput: +33.8%.
- Tamaño del motor: -39,5 %.

## Recursos y potencia

| Precisión | RAM media | CPU media | GPU media | Temp. GPU media | POM_5V_IN media | Energía de entrada |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 3151.9 MB | 28.9% | 73.4% | 64.8 C | 6.40 W | 147.8 J |
| FP16 | 2990.3 MB | 31.6% | 77.6% | 75.9 C | 7.47 W | 171.9 J |

Notas metodológicas:

- FP16 se ejecutó después de FP32; temperatura y potencia deben interpretarse con cautela por el efecto de orden y calentamiento.
- Los permisos de lectura de INA3221 se ajustaron temporalmente con `sudo chmod a+r` sobre los sensores IIO para poder muestrear potencia.
- En esta iteración no se repitió INT8. La cuantización calibrada se estudió después y se conserva
  en los README `jetson-domainmix-pd5-umair-hard-int8-results` y
  `jetson-domainmix-pd5-umair-hard-int8-50pc-results`.

## Evidencia local

- Logs de TensorRT: `F:\TFM experimento\jetson-domainmix-pd5-results\domainmix_pd5`.
- Resumen recursos: `F:\TFM experimento\jetson-domainmix-pd5-results\resource_summary.csv`.
- Resumen potencia: `F:\TFM experimento\jetson-domainmix-pd5-results\power_summary.csv`.
- Validación PlantDoc: `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_domainmix_pd5_from_e25_e10`.
- Validación Zenodo: `F:\TFM experimento\training\runs\external_eval\zenodo_val_domainmix_pd5_from_e25_e10`.
