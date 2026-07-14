# Notas de dataset

## Objetivo recomendado

Detectar hojas o lesiones de tomate afectadas por enfermedad en condiciones de campo.

## Clases iniciales sugeridas

- `tomato_healthy`
- `early_blight`
- `late_blight`
- `leaf_mold`
- `septoria_leaf_spot`
- `bacterial_spot`
- `target_spot`
- `mosaic_virus`
- `yellow_leaf_curl_virus`
- `spider_mites`

## Reglas de captura

- Usar varias plantas y varios dias.
- Incluir sol directo, sombra, contraluz y distintos fondos.
- Capturar hojas sanas y enfermas.
- Evitar que train/val/test compartan la misma planta o secuencia casi identica.
- Guardar metadatos cuando sea posible: fecha, ubicacion, camara, iluminacion y variedad.

## Reglas de etiquetado

Para deteccion:

- Usar bounding boxes ajustadas a la hoja afectada o lesion.
- Mantener criterio constante: si se etiqueta hoja completa, no mezclar con lesion puntual en la misma version del dataset.

Para severidad:

- Si se quiere estimar severidad, conviene segmentar lesiones o clasificar por niveles: leve, media, severa.

