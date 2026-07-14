# Resultados INT8 calibrado - pd5 + hard cases Umair

Fecha: 2026-05-14.

> **Alcance.** Este resultado corresponde al modelo del Hilo A con resolución 224. La mejora de
> latencia de INT8 frente a FP16 es específica de este motor y no se generaliza a los candidatos
> Pareto p03 y p07, donde INT8 no supera a FP16. PlantDoc val contiene 69 imágenes y participa en
> el desarrollo del modelo.

## Objetivo

Probar cuantización *post-training* INT8 calibrada para el modelo `pd5 + hard cases Umair` en Jetson Nano Advantech, y comparar precisión, latencia, recursos y consumo frente a FP32/FP16.

## Calibración

- Dataset de calibración: `F:\TFM experimento\training\dataset\calibration_int8_pd5_umair_hard_20pc`.
- Composición: 200 imágenes, 20 por clase, muestreadas del entrenamiento `domain_mix_zenodo_plantdoc_pd5_umair_hard`.
- Script de preparación: `F:\TFM experimento\training\scripts\prepare_int8_calibration_subset.py`.
- Script de build TensorRT: `F:\TFM experimento\training\scripts\build_trt_int8_engine.py`.
- Caché de calibración: `tomato_yolov8n_cls_domainmix_pd5_umair_hard_int8.cache`.

TensorRT completó la calibración en 70,526 s. Durante el build se registró la advertencia: `Int8 support requested on hardware without native Int8 support`. Por tanto, estos resultados son especialmente útiles para documentar el límite de la Jetson Nano Tegra X1 frente a plataformas con INT8 nativo.

## Validación funcional

Se evaluó el motor INT8 sobre PlantDoc val, con 69 imágenes en el espacio de etiquetas del modelo:

| Modelo / formato | PlantDoc val Top-1 |
|---|---:|
| pd5 + hard cases PT | 0,623 |
| pd5 + hard cases ONNX dinamico | 0,623 |
| pd5 + hard cases TensorRT INT8 calibrado | 0,565 |

La cuantización INT8 calibrada introduce una pérdida de 5,8 puntos porcentuales sobre PlantDoc val en esta primera configuración de calibración.

## Rendimiento Jetson

| Precisión | Motor | Throughput load | Latencia media | P99 | RAM media | POM_5V_IN media | Energía entrada |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 8 MB | 195,584 qps | 5,100 ms | 5,551 ms | 3150,4 MB | 6,09 W | 140,3 J |
| FP16 | 5 MB | 222,177 qps | 4,489 ms | 4,545 ms | 2948,9 MB | 7,15 W | 164,5 J |
| INT8 calibrado | 5 MB | 251,299 qps | 3,969 ms | 4,004 ms | 3107,3 MB | 5,78 W | 133,1 J |

## Lectura

- INT8 calibrado mejora latencia y throughput frente a FP32 y FP16 para este motor concreto.
- La precisión cae en PlantDoc val de 0,623 a 0,565, por lo que esta variante no se considera candidata final.
- La Jetson Nano Tegra X1 no tiene INT8 nativo; el resultado no debe generalizarse a otras plataformas.
- Se repitió INT8 con 50 imágenes por clase. No recuperó precisión y fue ligeramente más lento que la calibración de 20 imágenes por clase.

## Evidencia

- `build_pd5_umair_hard_int8.log`
- `domainmix_pd5_umair_hard_int8/resources/int8_trtexec_load.log`
- `domainmix_pd5_umair_hard_int8/resources/int8_tegrastats.log`
- `domainmix_pd5_umair_hard_int8/power/int8_ina3221.csv`
- `resource_summary.csv`
- `power_summary.csv`
- `eval/plantdoc_val_pd5_space_int8/summary.json`
- `eval/plantdoc_val_pd5_space_int8/predictions.csv`
