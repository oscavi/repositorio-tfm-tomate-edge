cd /home/mic-710ai/tfm-experimento
mkdir -p logs/domainmix_pd5_umair_hard_int8_50pc/resources logs/domainmix_pd5_umair_hard_int8_50pc/power
rm -f logs/domainmix_pd5_umair_hard_int8_50pc/resources/int8_50pc_tegrastats.log logs/domainmix_pd5_umair_hard_int8_50pc/resources/int8_50pc_trtexec_load.log logs/domainmix_pd5_umair_hard_int8_50pc/power/int8_50pc_ina3221.csv
tegrastats --interval 250 > logs/domainmix_pd5_umair_hard_int8_50pc/resources/int8_50pc_tegrastats.log &
tegra_pid=$!
python3 scripts/sample_jetson_power.py --out logs/domainmix_pd5_umair_hard_int8_50pc/power/int8_50pc_ina3221.csv --interval-ms 250 --duration 23 &
power_pid=$!
sleep 2
/usr/src/tensorrt/bin/trtexec --loadEngine=engines/tomato_yolov8n_cls_domainmix_pd5_umair_hard_int8_50pc.engine --warmUp=1000 --duration=20 --iterations=1000 2>&1 | tee logs/domainmix_pd5_umair_hard_int8_50pc/resources/int8_50pc_trtexec_load.log
kill $tegra_pid 2>/dev/null || true
wait $tegra_pid 2>/dev/null || true
wait $power_pid 2>/dev/null || true
