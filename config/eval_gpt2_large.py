# evaluate the base gpt2
# num_layers=36, num_heads=20, dim_model=1280
# 774M parameters
batch_size = 8
eval_iters = 500  # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = "gpt2-large"
