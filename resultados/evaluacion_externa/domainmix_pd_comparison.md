# Comparacion de pesos PlantDoc en adaptacion de dominio

Fecha: 2026-05-14.

Punto de partida comun: `zenodo_tomato_cls_fieldaug_e25\weights\best.pt`.

| Variante | PlantDoc repeat | Zenodo val Top-1 | PlantDoc val Top-1 | Confianza media PlantDoc | Lectura |
|---|---:|---:|---:|---:|---|
| FieldAug e25 | 0 | 0.985 | 0.290 | 0.889 | Modelo base antes de adaptacion |
| Domain mix pd1 | 1 | 0.979 | 0.536 | 0.681 | Mejora PlantDoc con perdida baja en Zenodo |
| Domain mix pd2 | 2 | 0.958 | 0.623 | 0.688 | Mejora PlantDoc, pero degrada Zenodo mas que pd5 |
| Domain mix pd5 | 5 | 0.984 | 0.623 | 0.808 | Mejor equilibrio observado |

Evaluacion externa adicional de `pd5`:

| Dataset externo | Imagenes evaluadas | Top-1 | Confianza media | Notas |
|---|---:|---:|---:|---|
| UmairPirzada valid, mapeado a clases pd5 | 6.431 | 0.879 | 0.957 | Conjunto adicional descrito como mezcla de laboratorio y escenas reales; se excluye `powdery_mildew` por no estar en el espacio de etiquetas de `pd5`. |

Evaluacion tras entrenamiento con hard cases Umair:

| Variante | Zenodo val Top-1 | PlantDoc val Top-1 | Umair valid Top-1 | Lectura |
|---|---:|---:|---:|---|
| Domain mix pd5 | 0.984 | 0.623 | 0.879 | Modelo principal anterior |
| Domain mix pd5 + Umair hard cases | 0.987 | 0.623 | 0.943 | Mejora el dominio Umair sin degradar Zenodo/PlantDoc |

Conclusion operativa:

- `pd5` se mantiene como candidato principal porque conserva Zenodo practicamente estable y alcanza el mismo Top-1 PlantDoc que `pd2`.
- `pd1` es una variante conservadora, util como evidencia de transicion, pero no alcanza el rendimiento externo de `pd2/pd5`.
- `pd2` no justifica sustituir a `pd5`, ya que empata en PlantDoc y cae a 0.958 en Zenodo.
- En UmairPirzada valid, `pd5` obtiene 0.879 Top-1. Este resultado es inferior a Zenodo y superior a PlantDoc, por lo que aporta una evaluacion intermedia de generalizacion.
- La variante entrenada con hard cases Umair mejora Umair valid hasta 0.943 y conserva PlantDoc en 0.623. En Jetson, mantiene rendimiento FP32 equivalente a `pd5`, pero su FP16 es mas lento que el FP16 anterior.

Siguiente paso recomendado:

- Mantener ambos modelos como candidatos: `pd5` por eficiencia FP16 y `pd5 + hard cases` por mejor generalizacion externa.
- Repetir mediciones alternando el orden FP32/FP16 para reducir sesgo termico.
- Probar cuantizacion INT8 calibrada como siguiente tecnica de optimizacion antes de aplicar poda estructural.
