# Resultados INT8 calibrado con 50 imágenes por clase - pd5 + hard cases Umair

Fecha: 2026-05-14.

> **Alcance.** Esta comparación pertenece al modelo del Hilo A con resolución 224. No contradice
> los resultados de p03 y p07, cuyos motores INT8 no mejoran la latencia de FP16. PlantDoc val se
> usa durante el desarrollo y no constituye un test final independiente.

## Objetivo

Comprobar si aumentar el conjunto de calibración INT8 de 20 a 50 imágenes por clase recupera precisión en PlantDoc val y mantiene las mejoras de latencia y consumo observadas en la primera cuantización.

## Calibración

- Dataset local: `F:\TFM experimento\training\dataset\calibration_int8_pd5_umair_hard_50pc`.
- Composición: 500 imágenes, 50 por clase.
- Script de preparación: `F:\TFM experimento\training\scripts\prepare_int8_calibration_subset.py`.
- Engine remoto: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_int8_50pc.engine`.
- Cache remoto: `/home/mic-710ai/tfm-experimento/engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_int8_50pc.cache`.

TensorRT completó la calibración en 80,934 s y volvió a registrar que la Jetson Nano Tegra X1 no tiene soporte INT8 nativo.

## Comparación INT8

| Variante | Imágenes de calibración | PlantDoc Top-1 | Throughput | Latencia | P99 | POM_5V_IN media | Energía entrada |
|---|---:|---:|---:|---:|---:|---:|---:|
| INT8 20/clase | 200 | 0,565 | 251,299 qps | 3,969 ms | 4,004 ms | 5,78 W | 133,1 J |
| INT8 50/clase | 500 | 0,565 | 241,760 qps | 4,125 ms | 4,166 ms | 6,02 W | 138,6 J |

## Lectura

- Aumentar el conjunto de calibración de 200 a 500 imágenes no recuperó precisión en PlantDoc val.
- La variante de 50 imágenes por clase fue ligeramente más lenta y consumió algo más que la variante de 20.
- Para este motor concreto, INT8 mantiene una mejora de latencia frente a FP16, pero con una pérdida de precisión que no se corrige ampliando moderadamente la calibración.
- Para el TFM, FP16 es la referencia más equilibrada; INT8 se mantiene como resultado exploratorio y dependiente del modelo y la plataforma.

## Evidencia

- `build_pd5_umair_hard_int8_50pc.log`
- `domainmix_pd5_umair_hard_int8_50pc/resources/int8_50pc_trtexec_load.log`
- `domainmix_pd5_umair_hard_int8_50pc/resources/int8_50pc_tegrastats.log`
- `domainmix_pd5_umair_hard_int8_50pc/power/int8_50pc_ina3221.csv`
- `resource_summary.csv`
- `power_summary.csv`
- `plantdoc_val_pd5_space_int8_50pc/summary.json`
- `plantdoc_val_pd5_space_int8_50pc/predictions.csv`
