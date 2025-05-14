idx=7
rank=4
echo "Idx: $idx"
#python forward_timing.py --model mlp --transform flatten  --idx $idx --rank $rank --bs 256
#python forward_timing.py --model mlp --transform mult_flatten --hidden_dim 8  --idx $idx --bs 2
#python forward_timing.py --model mlp --transform align  --idx $idx --rank $rank
#python forward_timing.py --model mlp --transform svals  --idx $idx --rank $rank 
#python forward_timing.py --model gl_mlp --transform pair --idx $idx --rank $rank
python forward_timing.py --model fast_gl_mlp --transform pair_fast --idx $idx --rank $rank --bs 128 --num_layers 2 --hidden_dim 16
