# Comparacion pd5 vs pd5 + hard cases Umair

Fecha: 2026-05-14.

## Objetivo

Comprobar si los errores detectados en el conjunto externo UmairPirzada pueden incorporarse como casos dificiles de entrenamiento sin degradar el rendimiento previo en Zenodo y PlantDoc.

## Analisis de errores de partida

Modelo analizado: `domainmix_pd5_from_fieldaug_e25_e10`.

Conjunto externo: `umair_valid_pd5_mapped`, 6.431 imagenes.

Resultados de error:

- Errores totales: 781.
- Casos correctos de baja confianza seleccionados: 89.
- Subconjunto hard cases generado: 870 imagenes.
- Ruta: `F:\TFM experimento\training\dataset\external_prepared\umair_valid_pd5_hard_cases`.

Principales confusiones:

| Etiqueta real | Prediccion | Errores |
|---|---|---:|
| healthy | Late blight | 92 |
| Leaf Mold | Late blight | 74 |
| Bacterial spot | Septoria leaf spot | 47 |
| Bacterial spot | Early blight | 46 |
| Septoria leaf spot | Late blight | 40 |
| Bacterial spot | Late blight | 38 |
| Septoria leaf spot | Early blight | 35 |
| healthy | Bacterial spot | 34 |

Artefactos:

- `F:\TFM experimento\training\runs\external_eval\umair_valid_pd5_error_analysis\class_error_summary.csv`
- `F:\TFM experimento\training\runs\external_eval\umair_valid_pd5_error_analysis\confusions_ranked.csv`
- `F:\TFM experimento\training\runs\external_eval\umair_valid_pd5_error_analysis\high_confidence_errors.jpg`

## Dataset de entrenamiento ampliado

Ruta:

`F:\TFM experimento\training\dataset\domain_mix_zenodo_plantdoc_pd5_umair_hard`

Construccion:

- Base: `domain_mix_zenodo_plantdoc_pd5`.
- Adicion: hard cases Umair con repeticion 2x en entrenamiento.
- Total entrenamiento: 15.125 imagenes.
- Validacion: PlantDoc val, 69 imagenes. Se debe interpretar con cautela porque solo contiene 8 de las 10 clases.

Script:

`F:\TFM experimento\training\scripts\prepare_domain_mix_hardcases.py`

## Entrenamiento

Run final:

`F:\TFM experimento\training\runs\classify\runs\domainmix_pd5_umair_hard_e25_w0`

Punto de partida:

`F:\TFM experimento\training\runs\classify\runs\domainmix_pd5_from_fieldaug_e25_e10\weights\best.pt`

Configuracion:

- Epocas solicitadas: 25.
- Parada temprana: epoca 11, por no mejorar la validacion interna durante 10 epocas.
- `workers=0` en Windows para evitar fallo de memoria compartida de PyTorch/DataLoader.
- Aumento de datos activado con los mismos parametros de simulacion de campo.

Nota tecnica:

El primer intento con `workers=8` fallo en la epoca 16 por `RuntimeError: Couldn't open shared file mapping`, error asociado a memoria compartida en Windows. Se repitio con `workers=0`.

## Evaluacion comparativa

| Modelo | Zenodo val Top-1 | PlantDoc val Top-1 | Umair valid Top-1 | Lectura |
|---|---:|---:|---:|---|
| pd5 | 0,984 | 0,623 | 0,879 | Modelo principal anterior. |
| pd5 + Umair hard cases | 0,987 | 0,623 | 0,943 | Mejora fuerte en Umair, conserva PlantDoc y Zenodo. |

Evaluaciones del nuevo modelo:

- Zenodo val: `F:\TFM experimento\training\runs\external_eval\umair_hard_e25_w0_zenodo_val`.
- PlantDoc val: `F:\TFM experimento\training\runs\external_eval\umair_hard_e25_w0_plantdoc_val`.
- Umair valid: `F:\TFM experimento\training\runs\external_eval\umair_hard_e25_w0_umair_valid`.

## Exportacion

Ruta:

`F:\TFM experimento\training\exports\domainmix_pd5_umair_hard_e25`

Ficheros:

- `tomato_yolov8n_cls_domainmix_pd5_umair_hard_e25.pt`
- `tomato_yolov8n_cls_domainmix_pd5_umair_hard_e25_dynamic.onnx`

Validacion ONNX dinamico:

- PlantDoc val Top-1: 0,623.
- Zenodo val Top-1: 0,986.

## Conclusion operativa

La incorporacion controlada de hard cases procedentes del conjunto externo mejora la generalizacion sobre ese dominio sin degradar las metricas externas principales disponibles.

## Despliegue comparativo en Jetson Nano

El modelo se exporto tambien a ONNX opset 12 para TensorRT:

- `F:\TFM experimento\training\exports\domainmix_pd5_umair_hard_e25\tomato_yolov8n_cls_domainmix_pd5_umair_hard_e25_opset12.onnx`

Resultados cargando engines ya compilados en la Jetson:

| Precision | Engine | Build | Throughput load | Latencia load | P99 load | RAM media | POM_5V_IN media | Energia entrada |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 8 MB | 102,974 s | 195,584 qps | 5,100 ms | 5,551 ms | 3150,4 MB | 6,09 W | 140,3 J |
| FP16 | 5 MB | 228,336 s | 222,177 qps | 4,489 ms | 4,545 ms | 2948,9 MB | 7,15 W | 164,5 J |

Comparado con `pd5`, FP32 mantiene una latencia practicamente identica. En FP16, sin embargo, el nuevo modelo queda por debajo del throughput anterior, aunque conserva la mejora de precision externa en Umair. Esto refuerza la necesidad de presentar el experimento como una comparacion precision-latencia-consumo, no solo como una mejora de accuracy.

Evidencia local:

- `F:\TFM experimento\jetson-domainmix-pd5-umair-hard-results\README.md`
- `F:\TFM experimento\jetson-domainmix-pd5-umair-hard-results\resource_summary.csv`
- `F:\TFM experimento\jetson-domainmix-pd5-umair-hard-results\power_summary.csv`

## Cuantizacion INT8 calibrada

Se preparo un subconjunto de calibracion con 200 imagenes, 20 por clase, y se genero un engine TensorRT INT8 calibrado en la Jetson.

| Formato | PlantDoc val Top-1 | Throughput load | Latencia load | P99 load | POM_5V_IN media |
|---|---:|---:|---:|---:|---:|
| PT / ONNX | 0,623 | - | - | - | - |
| TensorRT FP16 | - | 222,177 qps | 4,489 ms | 4,545 ms | 7,15 W |
| TensorRT INT8 calibrado | 0,565 | 251,299 qps | 3,969 ms | 4,004 ms | 5,78 W |

TensorRT aviso de que la Jetson Nano Tegra X1 no tiene soporte INT8 nativo. Aun asi, el engine INT8 calibrado mejora la latencia frente a FP16 en esta medicion, con una perdida de precision externa en PlantDoc val. La siguiente prueba deberia aumentar el conjunto de calibracion y evaluar tambien Zenodo/Umair antes de considerar esta variante como candidata.

Se repitio la calibracion con 50 imagenes por clase:

| Variante INT8 | Imagenes calibracion | PlantDoc val Top-1 | Throughput load | Latencia load | POM_5V_IN media |
|---|---:|---:|---:|---:|---:|
| 20/clase | 200 | 0,565 | 251,299 qps | 3,969 ms | 5,78 W |
| 50/clase | 500 | 0,565 | 241,760 qps | 4,125 ms | 6,02 W |

La ampliacion moderada del conjunto de calibracion no recupera precision y reduce ligeramente el rendimiento. En esta plataforma, INT8 queda como evidencia de trade-off y limitacion hardware, mientras que FP16/FP32 siguen siendo las variantes comparativas principales.

Evidencia local:

- `F:\TFM experimento\jetson-domainmix-pd5-umair-hard-int8-results\README.md`
- `F:\TFM experimento\jetson-domainmix-pd5-umair-hard-int8-results\eval\plantdoc_val_pd5_space_int8\summary.json`
- `F:\TFM experimento\jetson-domainmix-pd5-umair-hard-int8-50pc-results\README.md`

## Poda ligera de cabeza

Se probo una primera poda estructurada limitada a la cabeza de clasificacion, eliminando el 20% de sus canales:

- Canales de cabeza: 1280 -> 1024.
- Parametros: 1.451.098 -> 1.382.490.
- Reduccion total de parametros: 4,73%.

La variante podada conserva la precision local del modelo en PlantDoc antes de TensorRT:

| Variante | PlantDoc val Top-1 | Umair valid Top-1 |
|---|---:|---:|
| Base `pd5 + hard cases` | 0,623 | 0,943 |
| Head-prune 20% PT | 0,623 | 0,940 |

En Jetson Nano se compilo como TensorRT FP32 y FP16. La variante FP16 mejora claramente la latencia y el throughput:

| Variante | PlantDoc val Top-1 TensorRT | Throughput load | Latencia load | P99 load | POM_5V_IN media |
|---|---:|---:|---:|---:|---:|
| Base FP16 | - | 222,177 qps | 4,489 ms | 4,545 ms | 7,15 W |
| Head-prune 20% FP16 | 0,580 | 277,101 qps | 3,597 ms | 3,806 ms | 6,35 W |

La lectura es que la poda de cabeza aporta una mejora de rendimiento, pero el engine TensorRT pierde precision frente a la evaluacion local del `.pt`. Por tanto, es una variante adecuada para discutir el compromiso rendimiento-precision, no un reemplazo directo del modelo FP16 sin poda.

Evidencia local:

- `F:\TFM experimento\jetson-head-prune20-results\README.md`
- `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_e5\pruning_metadata.json`
- `F:\TFM experimento\training\runs\external_eval\head_prune20_init_plantdoc_val`

## Recuperacion corta tras poda

Se entreno una recuperacion controlada del modelo podado durante 5 epocas con learning rate bajo (`0,0001`) y augmentacion moderada.

| Variante | Zenodo val | PlantDoc val | Umair valid | PlantDoc TensorRT FP16 | Throughput FP16 | Latencia FP16 |
|---|---:|---:|---:|---:|---:|---:|
| Head-prune 20% sin recuperacion | - | 0,623 | 0,940 | 0,580 | 277,101 qps | 3,597 ms |
| Head-prune 20% recovery e5 | 0,986 | 0,623 | 0,945 | 0,565 | 277,185 qps | 3,595 ms |

La recuperacion corta mantiene el rendimiento y mejora ligeramente Umair valid en la evaluacion local, pero no recupera precision del engine TensorRT sobre PlantDoc. Por tanto, el resultado no desplaza al modelo FP16 sin poda como candidato principal.

Evidencia local:

- `F:\TFM experimento\jetson-head-prune20-recovery-results\README.md`
- `F:\TFM experimento\training\runs\pruning\head_prune20_recovery_torch_e5_lr1e4\finetune_metadata.json`
