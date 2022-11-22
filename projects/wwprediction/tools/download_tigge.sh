save_dir="data/weather_bench/baseline"
var_name="10m_v_component_of_wind"
python projects/wwprediction/tools/download_tigge.py ${var_name} --path ${save_dir}
