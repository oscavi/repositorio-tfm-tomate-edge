# Resultados TensorRT en Jetson Nano - TFM

Fecha de ejecución: 2026-05-13/14.

> **Alcance histórico.** Esta es la primera prueba TensorRT sobre el modelo e5. La variante INT8
> usa rangos sintéticos y no es una cuantización funcional comparable. Los resultados finales del
> TFM incorporan modelos posteriores, calibración INT8 real y poda. Todas las cifras de esta página
> son microbenchmarks del motor con entrada sintética, no tiempos de la aplicación completa.

## Hardware y entorno

- Equipo remoto: Advantech / Jetson Nano Developer Kit.
- IP: `192.168.1.81`.
- Usuario: `mic-710ai`.
- Sistema: Ubuntu 18.04.5 LTS.
- L4T: R32.6.1.
- JetPack: 4.6.
- CUDA: 10.2.
- TensorRT: 8.0.1.6.
- GPU: NVIDIA Tegra X1, compute capability 5.3.
- RAM: 3.9 GB.
- Disco raíz tras la prueba: 14 GB totales, 1.8 GB libres, 87 % usado.

## Modelo probado

- Caso de uso: clasificación de enfermedades en hojas de tomate.
- Arquitectura: `YOLOv8n-cls`.
- Entrenamiento: PC con RTX 3080.
- Dataset: Zenodo tomato leaf diseases, 10 clases.
- Variante desplegada: entrenamiento con aumentación de campo.
- Exactitud de validación PyTorch: top-1 `0.965`, top-5 `1.000`.
- Exactitud de validación ONNX opset 12 en PC: top-1 `0.966`, top-5 `1.000`.

Artefactos en PC:

- `F:\TFM experimento\training\runs\classify\runs\zenodo_tomato_cls_fieldaug_e5\weights\best.pt`
- `F:\TFM experimento\training\runs\classify\runs\zenodo_tomato_cls_fieldaug_e5\weights\best.onnx`

Artefactos en Jetson:

- `/home/mic-710ai/tfm-experimento/models/tomato_yolov8n_cls_fieldaug.pt`
- `/home/mic-710ai/tfm-experimento/models/tomato_yolov8n_cls_fieldaug_opset12.onnx`

## Conversión a TensorRT

Se usó `trtexec` desde:

```bash
/usr/src/tensorrt/bin/trtexec
```

El ONNX se exportó con opset 12 para compatibilidad con TensorRT 8.0.1 en JetPack 4.6.

## Resultados de rendimiento

Todas las pruebas usan entrada fija `1x3x224x224`, 500 ms de calentamiento y 15 s de medición.

| Precisión | Motor Jetson | Tamaño del motor | Build time | Throughput | Latencia media | P99 | GPU compute medio |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | `tomato_yolov8n_cls_fp32.engine` | 8.6 MB | 108.736 s | 196.655 qps | 5.07279 ms | 5.28027 ms | 5.01386 ms |
| FP16 | `tomato_yolov8n_cls_fp16.engine` | 5.3 MB | 234.145 s | 257.978 qps | 3.86486 ms | 4.06006 ms | 3.80346 ms |
| INT8 sintético | `tomato_yolov8n_cls_int8_synthetic.engine` | 8.2 MB | 189.093 s | 197.321 qps | 5.05523 ms | 5.27734 ms | 4.99675 ms |

## Lectura técnica

- FP16 es la mejor variante de esta prueba inicial para Jetson Nano.
- Frente a FP32, FP16 reduce la latencia media de `5.07279 ms` a `3.86486 ms`.
- La mejora relativa de latencia es aproximadamente `23.8%`.
- El throughput sube de `196.655 qps` a `257.978 qps`.
- La mejora relativa de throughput es aproximadamente `31.2%`.
- El engine FP16 baja de `8.6 MB` a `5.3 MB`.

## Nota sobre INT8

La prueba INT8 se conserva solo como evidencia técnica. TensorRT avisó:

```text
Int8 support requested on hardware without native Int8 support, performance will be negatively affected.
Calibrator is not being used. Users must provide dynamic range for all tensors that are not Int32.
```

Por tanto:

- Este INT8 no debe usarse como resultado final de cuantización.
- No es una medición válida de precisión.
- No mejora rendimiento en Jetson Nano porque esta plataforma no tiene soporte INT8 nativo comparable al de GPUs/Jetson mas recientes.
- La cuantización calibrada real se documenta en los README posteriores del modelo `pd5 + hard cases`.

## Logs preservados

- `trtexec_fp32.log`
- `trtexec_fp16.log`
- `trtexec_int8_synthetic.log`

## Comandos usados

FP32:

```bash
cd /home/mic-710ai/tfm-experimento
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/tomato_yolov8n_cls_fieldaug_opset12.onnx \
  --saveEngine=engines/tomato_yolov8n_cls_fp32.engine \
  --workspace=512 \
  --warmUp=500 \
  --duration=15 \
  --iterations=500
```

FP16:

```bash
cd /home/mic-710ai/tfm-experimento
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/tomato_yolov8n_cls_fieldaug_opset12.onnx \
  --saveEngine=engines/tomato_yolov8n_cls_fp16.engine \
  --fp16 \
  --workspace=512 \
  --warmUp=500 \
  --duration=15 \
  --iterations=500
```

INT8 sintético:

```bash
cd /home/mic-710ai/tfm-experimento
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/tomato_yolov8n_cls_fieldaug_opset12.onnx \
  --saveEngine=engines/tomato_yolov8n_cls_int8_synthetic.engine \
  --int8 \
  --workspace=512 \
  --warmUp=500 \
  --duration=15 \
  --iterations=500
```
