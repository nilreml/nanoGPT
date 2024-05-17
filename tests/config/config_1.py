out_dir = "/tmp/nanogpt/out-shakespeare-char"
eval_interval = 250
eval_iters = 20
log_interval = 20

always_save_checkpoint = False

wandb_log = False
wandb_project = "shakespeare-char"
wandb_run_name = "mini-gpt"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
seq_len = 64

bias = False

num_layers = 4
num_heads = 4
dim_model = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 20
lr_decay_iters = max_iters
min_lr = learning_rate / 10
beta2 = 0.95

warmup_iters = 10

compile = False

seed_offset = 0
