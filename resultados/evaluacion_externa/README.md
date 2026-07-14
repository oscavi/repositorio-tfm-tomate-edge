# Evaluación inicial de dominio con PlantDoc

Fecha: 2026-05-14

> **Alcance histórico.** Este README documenta la primera evaluación de PlantDoc, cuando sus
> imágenes de validación todavía no se habían utilizado para reentrenar el modelo. En el diseño
> completo del TFM, `PlantDoc train` se incorpora a la mezcla `pd5` y `PlantDoc val` interviene en
> la validación y selección de configuraciones. Por ello, los resultados posteriores se interpretan
> como adaptación y validación de dominio, no como un test externo final independiente.

## Objetivo

Evaluar el comportamiento de los modelos entrenados con Zenodo ante un conjunto distinto de
imágenes de tomate con condiciones menos controladas. En esta iteración inicial, PlantDoc se
utiliza como conjunto externo y todavía no interviene en el reentrenamiento.

## Preparación del dataset

Repositorio origen:

- `https://github.com/pratikkayal/PlantDoc-Dataset`

Ruta local del clon:

- `F:\TFM experimento\training\dataset\PlantDoc-Dataset`

Debido a que PlantDoc contiene nombres de archivo no válidos en Windows, se extrajo desde el índice Git saneando nombres:

- Script: `F:\TFM experimento\training\scripts\prepare_plantdoc_tomato.py`
- Salida: `F:\TFM experimento\training\dataset\plantdoc_tomato_cls`

Clases PlantDoc mapeadas:

| PlantDoc | Clase Zenodo/modelo |
|---|---|
| Tomato leaf bacterial spot | Tomato___Bacterial_spot |
| Tomato Early blight leaf | Tomato___Early_blight |
| Tomato leaf late blight | Tomato___Late_blight |
| Tomato mold leaf | Tomato___Leaf_Mold |
| Tomato Septoria leaf spot | Tomato___Septoria_leaf_spot |
| Tomato leaf yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| Tomato leaf mosaic virus | Tomato___Tomato_mosaic_virus |
| Tomato leaf | Tomato___healthy |
| Tomato two spotted spider mites leaf | Tomato___Spider_mites Two-spotted_spider_mite |

PlantDoc no proporciona una correspondencia directa para `Tomato___Target_Spot` en esta extracción.

## Tamaño del subconjunto

- Total extraído: 746 imágenes.
- Train mapeado: 677 imágenes.
- Val mapeado: 69 imágenes.
- La validación contiene 8 clases; *spider mites* aparece solo con 2 imágenes en train y no aparece en val.

## Resultados

| Modelo | Dataset | Imágenes | Top-1 | Confianza media |
|---|---:|---:|---:|---:|
| Baseline e5 | Zenodo val | 1.000 | 0.961 | - |
| Baseline e5 | PlantDoc tomate val | 69 | 0.304 | 0.844 |
| Field augmentation e5 | Zenodo val | 1.000 | 0.965 | - |
| Field augmentation e5 | PlantDoc tomate val | 69 | 0.319 | 0.839 |

## Interpretación

La caída de `0.965` a `0.319` en el modelo con aumentación indica un cambio de dominio importante entre el dataset de entrenamiento/validación original y las imágenes tipo campo de PlantDoc. La confianza media se mantiene alta (`0.839`), lo que sugiere sobreconfianza ante entradas fuera de la distribución aprendida.

La aumentación aplicada aporta una mejora pequeña frente al *baseline* (`0.319` frente a `0.304`), pero no resuelve el cambio de dominio. Para el experimento TFM esto permite justificar una fase posterior de adaptación de dominio, *fine-tuning* con datos de campo, calibración de confianza o evaluación de incertidumbre.

## Evidencias

- Baseline:
  - `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_baseline_e5\summary.json`
  - `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_baseline_e5\predictions.csv`
  - `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_baseline_e5\confusion_matrix.png`
- Augmentacion:
  - `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_fieldaug_e5\summary.json`
  - `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_fieldaug_e5\predictions.csv`
  - `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_fieldaug_e5\confusion_matrix.png`
  - `F:\TFM experimento\training\runs\external_eval\plantdoc_tomato_fieldaug_e5\prediction_examples.jpg`
