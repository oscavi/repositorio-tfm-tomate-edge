# Benchmark alternado Jetson Nano

Ensayo repetido y alternado con TensorRT FP16 para reducir sesgos de calentamiento, carga previa o deriva temporal. Cada tramo se ejecuto con `trtexec --loadEngine`, muestreo `tegrastats` y lectura INA3221.

## Secuencia ejecutada

| orden | variante | throughput qps | latencia media ms | P99 ms | RAM MB | CPU % | GPU % | W | J |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | base_fp16 | 221.955 | 4.495 | 4.550 | 2975.714 | 29.374 | 72.758 | 5.854 | 106.880 |
| 02 | prune_fp16 | 277.681 | 3.588 | 4.116 | 2975.374 | 31.665 | 74.165 | 5.976 | 107.615 |
| 03 | base_fp16 | 221.063 | 4.512 | 4.635 | 2975.286 | 29.563 | 73.407 | 6.037 | 110.165 |
| 04 | prune_recovery_fp16 | 278.138 | 3.583 | 3.769 | 2975.385 | 31.690 | 72.352 | 6.043 | 108.869 |
| 05 | base_fp16 | 221.188 | 4.509 | 4.708 | 2978.056 | 29.700 | 73.289 | 6.050 | 109.002 |

## Agregado por variante

| variante | n | throughput medio qps | sd qps | latencia media ms | sd ms | delta qps vs base | delta latencia vs base | potencia W | energia J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_fp16 | 3 | 221.402 | 0.483 | 4.505 | 0.009 | 0.000% | 0.000% | 5.980 | 108.682 |
| prune_fp16 | 1 | 277.681 |  | 3.588 |  | 25.419% | -20.361% | 5.976 | 107.615 |
| prune_recovery_fp16 | 1 | 278.138 |  | 3.583 |  | 25.626% | -20.472% | 6.043 | 108.869 |

## Lectura experimental

- La variante base FP16 es muy estable en las tres repeticiones: throughput medio cercano a 221 qps y latencia media alrededor de 4.51 ms.
- La poda ligera de cabeza aumenta el throughput aproximadamente un 25-26% y reduce la latencia alrededor de un 20%, manteniendo un consumo medio parecido en el rail de entrada.
- La version podada con recuperacion conserva la mejora de rendimiento, pero en la evaluacion externa TensorRT ya habia mostrado peor precision que el modelo base; por tanto se interpreta como alternativa de rendimiento, no como sustituto principal.
- Este benchmark complementa las metricas de precision: para la decision final debe cruzarse rendimiento, energia y exactitud externa PlantDoc/Umair.
