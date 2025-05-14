## LoL models for LLMs

See `main.py` for the main training script.

`scripts` contain basic scripts for training metanets and reproducing the experiments in the paper (e.g. `bash scripts/final_arc_c_test.sh`).

Download the data files `llama_arc.zip` and `qwen2_arc.zip` from (a source that we will share later). Move them both into the `lora_data/` directory, then unzip them.

`hparams.csv` contains the hyperparameters used for training the LLM LoRAs.
