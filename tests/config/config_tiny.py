out_dir = "/tmp/nanogpt/out-shakespeare-char"
eval_interval = 1050
eval_iters = 20
log_interval = 20

always_save_checkpoint = False

wandb_log = False
wandb_project = "shakespeare-char"
wandb_run_name = "mini-gpt"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 16
seq_len = 32

bias = False

num_layers = 4
num_heads = 1
dim_model = 32
dropout = 0.0

learning_rate = 1e-3
max_iters = 1000
lr_decay_iters = max_iters
min_lr = learning_rate / 10
beta2 = 0.95

warmup_iters = 10

compile = True

seed_offset = 0
