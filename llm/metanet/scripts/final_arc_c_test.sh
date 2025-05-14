# for qwen models
python main.py --lr 5e-4  --transform pair --model transformer --epochs 60 --target arc-c-test --repeats 3 --dataset qwen2_arc_data

# for llama models
#python main.py --lr 5e-5  --transform flatten --model mlp --epochs 60 --target arc-c-test --repeats 3 --dataset llama_arc_data
python main.py --lr 1e-4  --transform pair --model transformer --epochs 60 --target arc-c-test --repeats 3 --dataset llama_arc_data
#python main.py --lr 1e-4  --transform align --model mlp --epochs 60 --target arc-c-test --repeats 3 --dataset llama_arc_data   
#python main.py --lr 5e-4  --transform svals --model mlp --epochs 60 --target arc-c-test --repeats 3 --dataset llama_arc_data
#python main.py --lr 1e-3  --transform pair --model gl_mlp --epochs 60 --target arc-c-test --num_layers 2 --repeats 3 --dataset llama_arc_data
