# evaluate the base gpt2
# num_layers=12, num_heads=12, dim_model=768
# 124M parameters
batch_size = 8
eval_iters = 500  # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = "gpt2"
