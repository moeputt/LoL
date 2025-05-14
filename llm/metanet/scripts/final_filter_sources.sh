# for qwen models
python main.py --lr 5e-5  --transform pair --model transformer --epochs 60 --target filter_sources --repeats 3 --dataset qwen2_arc_data

# for llama models
python main.py --lr 1e-3  --transform pair --model transformer --epochs 60 --target filter_sources --repeats 3 --dataset llama_arc_data
