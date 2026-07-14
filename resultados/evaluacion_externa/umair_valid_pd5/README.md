# Evaluación inicial de dominio pd5: Umair-Pirzada Tomato Leaf Disease

Fecha de ejecución: 2026-05-14.

> **Alcance histórico.** Umair-Pirzada no se había usado para entrenar el modelo `pd5` en esta
> evaluación inicial. Después, el análisis de errores de este mismo conjunto se utiliza para
> seleccionar 870 *hard cases* y reentrenar. Por tanto, el valor inicial 0,8786 caracteriza la
> transferencia al nuevo dominio; las métricas posteriores caracterizan adaptación dirigida y no
> constituyen un test externo final independiente.

## Objetivo

Evaluar el modelo principal `pd5` sobre un dominio adicional de hojas de tomate que todavía no
había intervenido en su entrenamiento. El objetivo es medir la transferencia inicial, especialmente
ante imágenes descritas por la fuente como mezcla de laboratorio y escenas reales.

## Fuente

- Repositorio: https://github.com/UmairPirzada/Tomato-leaf-disease-classification
- Descripción de la fuente: clasificación de enfermedades de hoja de tomate con más de 20.000 imágenes, 10 enfermedades y 1 clase sana; el README indica mezcla de escenas de laboratorio y entorno real.
- Dataset enlazado por la fuente: https://www.kaggle.com/datasets/ashishmotwani/tomato
- Descarga realizada: clon sparse del repositorio GitHub, limitado a `Tomato leaf disease Dataset/valid`.

## Preparación local

Ruta fuente local:

`F:\TFM experimento\training\dataset\external_candidates\umair_tomato_leaf_disease\Tomato leaf disease Dataset\valid`

Ruta preparada:

`F:\TFM experimento\training\dataset\external_prepared\umair_valid_pd5_mapped`

Script de preparación:

`F:\TFM experimento\training\scripts\prepare_umair_external.py`

El script mapea las clases del nuevo dominio al espacio de etiquetas del modelo `pd5`. La clase `powdery_mildew` se excluye porque el modelo principal no fue entrenado con esa categoría.

## Distribución preparada

| Clase fuente | Clase pd5 | Imágenes | Estado |
|---|---|---:|---|
| Bacterial_spot | Tomato___Bacterial_spot | 732 | incluida |
| Early_blight | Tomato___Early_blight | 643 | incluida |
| healthy | Tomato___healthy | 805 | incluida |
| Late_blight | Tomato___Late_blight | 792 | incluida |
| Leaf_Mold | Tomato___Leaf_Mold | 739 | incluida |
| powdery_mildew | - | 252 | excluida |
| Septoria_leaf_spot | Tomato___Septoria_leaf_spot | 746 | incluida |
| Spider_mites Two-spotted_spider_mite | Tomato___Spider_mites Two-spotted_spider_mite | 435 | incluida |
| Target_Spot | Tomato___Target_Spot | 457 | incluida |
| Tomato_mosaic_virus | Tomato___Tomato_mosaic_virus | 584 | incluida |
| Tomato_Yellow_Leaf_Curl_Virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 498 | incluida |

Total evaluado: 6.431 imágenes.

## Modelo evaluado

`F:\TFM experimento\training\exports\domainmix_pd5\tomato_yolov8n_cls_domainmix_pd5_from_e25.pt`

Comando:

```powershell
& 'F:\TFM experimento\training\.venv\Scripts\python.exe' 'F:\TFM experimento\training\scripts\eval_cls_external.py' --model 'F:\TFM experimento\training\exports\domainmix_pd5\tomato_yolov8n_cls_domainmix_pd5_from_e25.pt' --data 'F:\TFM experimento\training\dataset\external_prepared\umair_valid_pd5_mapped' --out 'F:\TFM experimento\training\runs\external_eval\umair_valid_pd5' --imgsz 224 --batch 64
```

## Resultados

- Top-1 accuracy: 0,8786
- Aciertos: 5.650 / 6.431
- Confianza media: 0,9570

| Clase | Imágenes | Aciertos | Accuracy |
|---|---:|---:|---:|
| Tomato___Bacterial_spot | 732 | 568 | 0,7760 |
| Tomato___Early_blight | 643 | 610 | 0,9487 |
| Tomato___Late_blight | 792 | 727 | 0,9179 |
| Tomato___Leaf_Mold | 739 | 623 | 0,8430 |
| Tomato___Septoria_leaf_spot | 746 | 601 | 0,8056 |
| Tomato___Spider_mites Two-spotted_spider_mite | 435 | 412 | 0,9471 |
| Tomato___Target_Spot | 457 | 416 | 0,9103 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 498 | 496 | 0,9960 |
| Tomato___Tomato_mosaic_virus | 584 | 561 | 0,9606 |
| Tomato___healthy | 805 | 636 | 0,7901 |

## Artefactos generados

- `summary.json`: métricas agregadas.
- `predictions.csv`: predicción por imagen.
- `confusion_matrix.png`: matriz de confusión.
- `prediction_examples.jpg`: ejemplos visuales de predicción.

## Observaciones

El rendimiento global cae respecto a Zenodo/PlantVillage y se mantiene por encima de PlantDoc, lo que sugiere que este conjunto tiene una dificultad intermedia para el modelo. Las clases con mayor degradación relativa son `Bacterial_spot`, `healthy` y `Septoria_leaf_spot`. El análisis posterior selecciona ejemplos similares para la fase de adaptación de dominio; desde ese momento, Umair deja de funcionar como evaluación independiente.
