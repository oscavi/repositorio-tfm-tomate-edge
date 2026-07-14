# Resultados Jetson - domain mix pd5 + hard cases Umair

Fecha: 2026-05-14.

> **Alcance.** Los *hard cases* proceden de `Umair valid`; por tanto, el paso de 0,879 a 0,943
> mide adaptación dirigida al dominio Umair, no generalización sobre un test independiente. Las
> métricas de `trtexec` corresponden al motor TensorRT, no a una aplicación extremo a extremo.

## Objetivo

Comparar en Jetson Nano Advantech el modelo `domain mix pd5 + hard cases Umair` frente al modelo `domain mix pd5` anterior, midiendo validación por dominios, coste de compilación TensorRT, latencia, throughput, uso de recursos, temperatura y consumo eléctrico aproximado.

## Artefactos evaluados

- Modelo PT local: `F:\TFM experimento\training\exports\domainmix_pd5_umair_hard_e25\tomato_yolov8n_cls_domainmix_pd5_umair_hard_e25.pt`.
- ONNX TensorRT local: `F:\TFM experimento\training\exports\domainmix_pd5_umair_hard_e25\tomato_yolov8n_cls_domainmix_pd5_umair_hard_e25_opset12.onnx`.
- ONNX remoto: `/home/mic-710ai/tfm-experimento/models/tomato_yolov8n_cls_domainmix_pd5_umair_hard_e25_opset12.onnx`.
- Engine FP32 remoto: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_fp32.engine`.
- Engine FP16 remoto: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_fp16.engine`.

## Validación por dominios

| Modelo | Zenodo val Top-1 | PlantDoc val Top-1 | Umair valid Top-1 |
|---|---:|---:|---:|
| Domain mix pd5 | 0,984 | 0,623 | 0,879 |
| pd5 + hard cases Umair | 0,987 | 0,623 | 0,943 |

## TensorRT

| Precisión | Motor | Build | Throughput build | Latencia build | Throughput load | Latencia load | P99 load |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 8 MB | 102,974 s | 196,963 qps | 5,065 ms | 195,584 qps | 5,100 ms | 5,551 ms |
| FP16 | 5 MB | 228,336 s | 221,232 qps | 4,508 ms | 222,177 qps | 4,489 ms | 4,545 ms |

## Recursos y potencia

Medición con `tegrastats` y muestreo INA3221. La duración efectiva de los CSV de potencia fue de unos 23 s por precisión.

| Precisión | RAM media | CPU media | GPU media | Temp. GPU media | POM_5V_IN media | Energía de entrada |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 3150,4 MB | 28,6 % | 68,5 % | 65,1 C | 6,09 W | 140,3 J |
| FP16 | 2948,9 MB | 30,2 % | 76,6 % | 72,8 C | 7,15 W | 164,5 J |

## Comparación frente a pd5 anterior

| Modelo | Precisión | Umair Top-1 | Throughput load | Latencia load | RAM media | POM_5V_IN media | Energía de entrada |
|---|---|---:|---:|---:|---:|---:|---:|
| pd5 | FP32 | 0,879 | 195,418 qps | 5,105 ms | 3151,9 MB | 6,40 W | 147,8 J |
| pd5 + hard cases | FP32 | 0,943 | 195,584 qps | 5,100 ms | 3150,4 MB | 6,09 W | 140,3 J |
| pd5 | FP16 | 0,879 | 261,452 qps | 3,814 ms | 2990,3 MB | 7,47 W | 171,9 J |
| pd5 + hard cases | FP16 | 0,943 | 222,177 qps | 4,489 ms | 2948,9 MB | 7,15 W | 164,5 J |

## Lectura

- La adición de *hard cases* mejora la adaptación a Umair valid, de 0,879 a 0,943 Top-1.
- En FP32 el coste de inferencia en Jetson se mantiene prácticamente igual que en `pd5`.
- En FP16 el nuevo motor es más lento que el FP16 anterior: 222,177 qps frente a 261,452 qps. La mejora de adaptación no viene gratis en esta configuración.
- El consumo medio de entrada queda en el mismo orden de magnitud que el modelo anterior. La comparación energética debe interpretarse con cautela porque las mediciones se ejecutan secuencialmente y la temperatura inicial no es idéntica.
- Esta limitación motivó los experimentos posteriores de cuantización calibrada y el protocolo
  repetido e intercalado de base, poda y recuperación.

## Evidencia local

- `trtexec_domainmix_pd5_umair_hard_fp32.log`
- `trtexec_domainmix_pd5_umair_hard_fp16.log`
- `domainmix_pd5_umair_hard/resources/fp32_trtexec_load.log`
- `domainmix_pd5_umair_hard/resources/fp16_trtexec_load.log`
- `domainmix_pd5_umair_hard/resources/fp32_tegrastats.log`
- `domainmix_pd5_umair_hard/resources/fp16_tegrastats.log`
- `domainmix_pd5_umair_hard/power/fp32_ina3221.csv`
- `domainmix_pd5_umair_hard/power/fp16_ina3221.csv`
- `resource_summary.csv`
- `power_summary.csv`
