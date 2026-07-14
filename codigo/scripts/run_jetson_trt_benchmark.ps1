param(
    [string]$JetsonHost = "192.168.1.81",
    [string]$JetsonUser = "mic-710ai",
    [string]$JetsonPassword = "mic-710ai",
    [string]$HostKey = "SHA256:0Y0cDc1K8XFOrJ+RbO7np4+2QCqDDAJy87myGTBVkwU",
    [string]$OnnxPath = "F:\TFM experimento\training\runs\classify\runs\zenodo_tomato_cls_fieldaug_e5\weights\best.onnx",
    [string]$PtPath = "F:\TFM experimento\training\runs\classify\runs\zenodo_tomato_cls_fieldaug_e5\weights\best.pt",
    [string]$LocalResultsDir = "F:\TFM experimento\jetson-trt-results"
)

$ErrorActionPreference = "Stop"

$plink = "C:\Program Files\PuTTY\plink.exe"
$pscp = "C:\Program Files\PuTTY\pscp.exe"
$remote = "$JetsonUser@$JetsonHost"
$remoteRoot = "/home/$JetsonUser/tfm-experimento"

New-Item -ItemType Directory -Force $LocalResultsDir | Out-Null

& $plink -batch -hostkey $HostKey -ssh -pw $JetsonPassword $remote "mkdir -p $remoteRoot/{models,engines,logs,reports,scripts,data}"
& $pscp -batch -hostkey $HostKey -pw $JetsonPassword $OnnxPath "${remote}:$remoteRoot/models/tomato_yolov8n_cls_fieldaug_opset12.onnx"
& $pscp -batch -hostkey $HostKey -pw $JetsonPassword $PtPath "${remote}:$remoteRoot/models/tomato_yolov8n_cls_fieldaug.pt"

$cmdFp32 = "cd $remoteRoot && /usr/src/tensorrt/bin/trtexec --onnx=models/tomato_yolov8n_cls_fieldaug_opset12.onnx --saveEngine=engines/tomato_yolov8n_cls_fp32.engine --workspace=512 --warmUp=500 --duration=15 --iterations=500 2>&1 | tee logs/trtexec_fp32.log"
$cmdFp16 = "cd $remoteRoot && /usr/src/tensorrt/bin/trtexec --onnx=models/tomato_yolov8n_cls_fieldaug_opset12.onnx --saveEngine=engines/tomato_yolov8n_cls_fp16.engine --fp16 --workspace=512 --warmUp=500 --duration=15 --iterations=500 2>&1 | tee logs/trtexec_fp16.log"
$cmdInt8 = "cd $remoteRoot && /usr/src/tensorrt/bin/trtexec --onnx=models/tomato_yolov8n_cls_fieldaug_opset12.onnx --saveEngine=engines/tomato_yolov8n_cls_int8_synthetic.engine --int8 --workspace=512 --warmUp=500 --duration=15 --iterations=500 2>&1 | tee logs/trtexec_int8_synthetic.log"

& $plink -batch -hostkey $HostKey -ssh -pw $JetsonPassword $remote $cmdFp32
& $plink -batch -hostkey $HostKey -ssh -pw $JetsonPassword $remote $cmdFp16
& $plink -batch -hostkey $HostKey -ssh -pw $JetsonPassword $remote $cmdInt8

& $pscp -batch -hostkey $HostKey -pw $JetsonPassword "${remote}:$remoteRoot/logs/trtexec_fp32.log" "$LocalResultsDir\trtexec_fp32.log"
& $pscp -batch -hostkey $HostKey -pw $JetsonPassword "${remote}:$remoteRoot/logs/trtexec_fp16.log" "$LocalResultsDir\trtexec_fp16.log"
& $pscp -batch -hostkey $HostKey -pw $JetsonPassword "${remote}:$remoteRoot/logs/trtexec_int8_synthetic.log" "$LocalResultsDir\trtexec_int8_synthetic.log"

& $plink -batch -hostkey $HostKey -ssh -pw $JetsonPassword $remote "cd $remoteRoot && ls -lh models engines logs && grep -H 'Throughput:\|Latency: min\|GPU Compute Time:\|Engine built in\|Loaded engine size\|Int8 support requested\|Calibrator is not being used' logs/trtexec_*.log"
