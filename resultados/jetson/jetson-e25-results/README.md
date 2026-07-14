# Iteración e25: entrenamiento, evaluación inicial de dominio y despliegue Jetson

Fecha: 2026-05-14

> **Alcance histórico.** PlantDoc se mantiene separado del entrenamiento en esta iteración e25.
> Después se incorpora `PlantDoc train` a la mezcla pd5 y `PlantDoc val` interviene en la
> selección. Por tanto, esta evaluación inicial evidencia el cambio de dominio, pero PlantDoc no
> funciona como test final independiente en el conjunto del TFM. Las métricas de Jetson proceden
> de un microbenchmark del motor TensorRT.

## Entrenamiento en PC

Dataset de entrenamiento:

- `F:\TFM experimento\training\dataset\zenodo_tomato_leaf\tomato-dataset`

Modelos entrenados:

| Modelo | Épocas solicitadas | Épocas completadas | Top-1 Zenodo val | Top-5 Zenodo val |
|---|---:|---:|---:|---:|
| `zenodo_tomato_cls_baseline_e25` | 25 | 25 | 0.987 | 1.000 |
| `zenodo_tomato_cls_fieldaug_e25` | 25 | 21 | 0.985 | 1.000 |

Nota: la variante con aumentación aplicó *early stopping* tras 21 épocas. El mejor checkpoint se observó alrededor de la época 11 y se conservó como `best.pt`.

## Evaluación inicial con PlantDoc

PlantDoc se usó como conjunto externo en esta iteración concreta, sin reentrenamiento.

| Modelo | Imágenes PlantDoc | Top-1 PlantDoc | Confianza media |
|---|---:|---:|---:|
| Baseline e25 | 69 | 0.275 | 0.895 |
| Field augmentation e25 | 69 | 0.290 | 0.889 |

Interpretación:

- El aumento de épocas mejora la validación interna Zenodo, pero no mejora PlantDoc frente a e5.
- Esto refuerza la evidencia de cambio de dominio y posible sobreajuste al dataset original.
- El modelo mantiene confianza media alta en PlantDoc pese a la baja precisión, lo que sugiere sobreconfianza ante entradas fuera de distribución.

## Exportación

Modelo exportado:

- Origen: `F:\TFM experimento\training\runs\classify\runs\zenodo_tomato_cls_fieldaug_e25\weights\best.pt`
- ONNX local: `F:\TFM experimento\training\exports\e25\tomato_yolov8n_cls_fieldaug_e25_opset12.onnx`
- ONNX remoto: `/home/mic-710ai/tfm-experimento/models/tomato_yolov8n_cls_fieldaug_e25_opset12.onnx`

Validación ONNX:

- Top-1 Zenodo val: 0.986.
- Top-5 Zenodo val: 1.000.
- Entrada: `images [1, 3, 224, 224]`.
- Salida: `output0 [1, 10]`.

## TensorRT en Jetson

Dispositivo:

- Jetson Nano / Tegra X1.
- TensorRT 8.0.1.
- JetPack 4.6.

| Precisión | Motor remoto | Tamaño | Build | Throughput | Latencia media | P99 | GPU compute medio |
|---|---|---:|---:|---:|---:|---:|---:|
| FP32 | `tomato_yolov8n_cls_fieldaug_e25_fp32.engine` | 8.6 MB | 101.155 s | 173.694 qps | 5.745 ms | 6.025 ms | 5.686 ms |
| FP16 | `tomato_yolov8n_cls_fieldaug_e25_fp16.engine` | 5.3 MB | 229.506 s | 259.594 qps | 3.840 ms | 4.170 ms | 3.779 ms |

FP16 frente a FP32:

- Reduccion de latencia media: 33.2%.
- Aumento de throughput: 49.5%.
- Reducción del tamaño del motor: 38,4 %.

## Recursos y potencia

Medición con motores ya construidos, usando `--loadEngine`, 20 s por precisión y muestreo cada 250 ms.

| Precisión | RAM media | CPU media | GPU media | Temp. GPU media | POM_5V_IN media | POM_5V_GPU media | POM_5V_CPU media | Energía de entrada |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 3160.5 MB | 27.8% | 73.2% | 66.8 C | 6.35 W | 2.50 W | 1.32 W | 146.3 J |
| FP16 | 2983.9 MB | 31.6% | 76.6% | 78.2 C | 7.60 W | 3.16 W | 1.81 W | 174.7 J |

Nota metodológica:

- FP16 se ejecutó después de FP32, por lo que temperatura y potencia pueden estar afectadas por el estado térmico acumulado.
- Para una comparación energética formal conviene repetir en orden aleatorizado, con enfriamiento y varias rondas.

## Evidencias locales

- Logs TensorRT: `F:\TFM experimento\jetson-e25-results\trtexec_fp32.log`, `trtexec_fp16.log`.
- Logs de carga: `F:\TFM experimento\jetson-e25-results\fp32_trtexec_load.log`, `fp16_trtexec_load.log`.
- Recursos: `F:\TFM experimento\jetson-e25-results\resource_summary.csv`.
- Potencia: `F:\TFM experimento\jetson-e25-results\power_summary.csv`.
- Evaluación inicial PlantDoc e25: `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_fieldaug_e25`.
