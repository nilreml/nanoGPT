out_dir = "/tmp/nanogpt/out-shakespeare-char"
eval_interval = 250
eval_iters = 20
log_interval = 1

always_save_checkpoint = False

wandb_log = False
wandb_project = "shakespeare-char"
wandb_run_name = "mini-gpt"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 256
seq_len = 64

bias = False

num_layers = 16
num_heads = 8
dim_model = 256
dropout = 0.0

learning_rate = 2e-3
max_iters = 500
lr_decay_iters = max_iters
min_lr = learning_rate / 10
beta2 = 0.95

warmup_iters = 80

compile = True

seed_offset = 30
