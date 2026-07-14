# Instrumentación de potencia en Jetson

Fecha: 2026-05-14

> **Alcance histórico.** Este README conserva la primera prueba de instrumentación. La memoria
> utiliza después un protocolo repetido e intercalado para comparar base, poda y recuperación.
> INA3221 estima potencia de placa y permite comparaciones homogéneas, pero no sustituye un
> medidor externo calibrado ni caracteriza el consumo completo de una aplicación con cámara.

## Hallazgo principal

La Jetson no muestra potencia en la salida básica de `tegrastats`, pero sí expone un sensor interno `ina3221x` mediante sysfs/IIO:

`/sys/bus/iio/devices/iio:device0`

Raíles detectados:

- `POM_5V_IN`
- `POM_5V_GPU`
- `POM_5V_CPU`

Las lecturas requieren permisos de root. Las unidades observadas son mV, mA y mW.

## Utilidades instaladas o disponibles

- `i2c-tools` ya estaba instalado (`i2cdetect`, `i2cget`).
- `tegrastats` ya estaba disponible por JetPack.
- No se instaló software adicional porque el sensor eléctrico ya era accesible por sysfs y conviene conservar espacio en disco.
- `sysstat`, `powertop`, `powerstat` y `lm-sensors` no estaban instalados. Pueden instalarse, pero no son necesarios para leer los railes INA3221 en esta prueba.
- `apt-cache policy` confirma candidatos disponibles en Ubuntu 18.04 arm64:
  - `sysstat` 11.6.1-1ubuntu0.2.
  - `lm-sensors` 1:3.4.0-4ubuntu0.1.
  - `powertop` 2.9-0ubuntu1.
  - `powerstat` 0.02.15-1.

## Scripts creados

- `F:\TFM experimento\training\scripts\sample_jetson_power.py`
- `F:\TFM experimento\training\scripts\parse_power_samples.py`
- `F:\TFM experimento\training\scripts\run_power_benchmark_jetson.sh`

Copias remotas en la Jetson:

- `/home/mic-710ai/tfm-experimento/scripts/sample_jetson_power.py`
- `/home/mic-710ai/tfm-experimento/scripts/run_power_benchmark_jetson.sh`

## Resultados preliminares

Duración de carga por precisión: 20 s de `trtexec --loadEngine`, con muestreo cada 250 ms y margen de captura.

| Precisión | POM_5V_IN media | POM_5V_IN p95 | POM_5V_GPU media | POM_5V_CPU media | Energía de entrada estimada |
|---|---:|---:|---:|---:|---:|
| FP32 | 6.17 W | 8.47 W | 2.37 W | 1.35 W | 142.0 J |
| FP16 | 6.86 W | 8.83 W | 2.69 W | 1.66 W | 157.9 J |

## Interpretación

Estos datos sirven como instrumentación inicial del consumo eléctrico interno. La comparación FP32/FP16 no debe interpretarse como definitiva, porque las ejecuciones fueron secuenciales y FP16 se midió después de FP32, con estado térmico y de frecuencia potencialmente distinto.

Para una comparación final se recomienda:

1. Repetir varias rondas.
2. Alternar el orden FP32/FP16.
3. Dejar enfriar entre ejecuciones.
4. Registrar simultaneamente `tegrastats`.
5. Contrastar con un medidor externo en la entrada de alimentacion si se requiere potencia absoluta.
