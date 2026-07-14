# Benchmark alternado Jetson Nano

Ensayo repetido y alternado con TensorRT FP16 para reducir sesgos de calentamiento, carga previa o deriva temporal. Cada tramo se ejecuto con `trtexec --loadEngine`, muestreo `tegrastats` y lectura INA3221.

## Secuencia ejecutada

| orden | variante | throughput qps | latencia media ms | P99 ms | RAM MB | CPU % | GPU % | W | J |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | base_fp16 | 221.398 | 4.505 | 4.572 | 2975.615 | 29.560 | 72.956 | 5.936 | 107.031 |
| 02 | prune_fp16 | 276.997 | 3.597 | 3.911 | 2971.857 | 31.764 | 72.176 | 6.122 | 110.411 |
| 03 | prune_recovery_fp16 | 274.781 | 3.620 | 4.159 | 2976.934 | 32.025 | 70.758 | 6.151 | 110.751 |
| 04 | base_fp16 | 220.528 | 4.522 | 4.759 | 2975.154 | 30.052 | 72.462 | 6.163 | 110.926 |
| 05 | prune_fp16 | 277.164 | 3.596 | 3.905 | 2978.315 | 32.292 | 72.674 | 6.376 | 114.911 |
| 06 | prune_recovery_fp16 | 276.222 | 3.606 | 3.968 | 2978.231 | 32.184 | 73.615 | 6.264 | 112.735 |
| 07 | base_fp16 | 220.232 | 4.528 | 4.773 | 2973.626 | 30.464 | 73.813 | 6.223 | 112.217 |
| 08 | prune_fp16 | 276.951 | 3.598 | 3.918 | 2978.176 | 32.014 | 73.780 | 6.302 | 113.580 |
| 09 | prune_recovery_fp16 | 276.057 | 3.608 | 3.943 | 2976.703 | 31.942 | 73.582 | 6.276 | 113.124 |

## Agregado por variante

| variante | n | throughput medio qps | sd qps | latencia media ms | sd ms | delta qps vs base | delta latencia vs base | potencia W | energia J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_fp16 | 3 | 220.719 | 0.606 | 4.518 | 0.012 | 0.000% | 0.000% | 6.107 | 110.058 |
| prune_fp16 | 3 | 277.037 | 0.112 | 3.597 | 0.001 | 25.516% | -20.391% | 6.267 | 112.967 |
| prune_recovery_fp16 | 3 | 275.687 | 0.789 | 3.611 | 0.008 | 24.904% | -20.074% | 6.230 | 112.203 |

## Lectura experimental

- La variante base FP16 es muy estable en las tres repeticiones: throughput medio cercano a 221 qps y latencia media alrededor de 4.51 ms.
- La poda ligera de cabeza aumenta el throughput aproximadamente un 25-26% y reduce la latencia alrededor de un 20%, manteniendo un consumo medio parecido en el rail de entrada.
- La version podada con recuperacion conserva la mejora de rendimiento, pero en la evaluacion externa TensorRT ya habia mostrado peor precision que el modelo base; por tanto se interpreta como alternativa de rendimiento, no como sustituto principal.
- Este benchmark complementa las metricas de precision: para la decision final debe cruzarse rendimiento, energia y exactitud externa PlantDoc/Umair.
