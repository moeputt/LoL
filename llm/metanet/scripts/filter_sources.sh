dataset=qwen2_arc_data
#dataset=llama_arc_data
lrs=(5e-5 1e-4 5e-4 1e-3 5e-3)
for lr in "${lrs[@]}"; do
    #python main.py --lr $lr  --transform flatten --model mlp --epochs 60 --target filter_sources --dataset $dataset
    #python main.py --lr $lr  --transform mult_flatten  --model mlp --epochs 60 --bs 8 --hidden_dim 16 --target filter_sources
    #python main.py --lr $lr  --transform align --model mlp --epochs 60 --target filter_sources --dataset $dataset
    #python main.py --lr $lr  --transform svals --model mlp --epochs 60 --target filter_sources --dataset $dataset
    #python main.py --lr $lr  --transform pair --model gl_mlp --epochs 60 --target filter_sources --num_layers 2 --dataset $dataset
    #python main.py --lr $lr  --transform pair --model 1layer_gl_mlp --epochs 60 --target filter_sources --num_layers 1 --dataset $dataset
    python main.py --lr $lr  --transform pair --model transformer --epochs 60 --target filter_sources --hidden_dim 128 --dataset $dataset
done

