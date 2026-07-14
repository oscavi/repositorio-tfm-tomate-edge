# Barrido multiobjetivo de hiperparámetros

Barrido exploratorio de 12 configuraciones para la clasificación de enfermedades foliares del
tomate. Se modifican resolución, *learning rate*, *weight decay*, *dropout*, *scheduler* coseno,
perfil de aumentación, MixUp/CutMix, congelación de capas y optimizador.

## Objetivos

La primera fase maximiza la exactitud Top-1 en Zenodo, PlantDoc y Umair. Después se incorporan
las mediciones de latencia, throughput y potencia de los candidatos ejecutados en Jetson.

## Alcance metodológico

- Zenodo es el dominio base; PlantDoc es un dominio intermedio de adaptación y validación; Umair
  es el dominio objetivo utilizado también para minería de *hard cases*.
- Los tres conjuntos son dominios de validación dentro del desarrollo. PlantDoc y Umair no forman
  un test externo final completamente independiente.
- El barrido utiliza una semilla por configuración y PlantDoc val contiene 69 imágenes. Las
  diferencias pequeñas y la pertenencia al frente deben interpretarse como exploratorias.
- En tres objetivos, 10 de 12 configuraciones son no dominadas; solo p06 y p12 quedan dominadas.
  Las estrellas p02, p03, p05, p07 y p08 representan el subconjunto operativo llevado a Jetson,
  no la totalidad del frente de Pareto matemático.
- La selección entre p02 y p03 es provisional: sus exactitudes disponibles proceden de PyTorch y
  falta validarlas después de la conversión a TensorRT sobre las mismas imágenes.

## Procedencia de la ejecución

- Checkpoint base original: `F:\TFM experimento\training\runs\classify\runs\domainmix_pd5_from_fieldaug_e25_e10\weights\best.pt`.
- Dataset de entrenamiento original: `F:\TFM experimento\training\dataset\domain_mix_zenodo_plantdoc_pd5_umair_hard`.
- Código público: `codigo/scripts/run_pareto_sweep.py` y `codigo/scripts/analyze_pareto_results.py`.
- Resultados curados: este directorio y `resultados/jetson/jetson_pareto_candidates/`.
