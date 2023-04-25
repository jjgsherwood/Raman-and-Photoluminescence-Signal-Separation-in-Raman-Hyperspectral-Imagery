NUM_WORKERS = 0
VALIDATION_PER = 0.1

# determine after how many epochs the training phases switch.
PRETRAINING_PHASE_1 = 5
PRETRAINING_PHASE_2 = 8

NN_INPUTS = {
    'cuda': True,
    'log_step': 700,
    'lr': 0.001
}

# show extra plots
SHOW_WEIGHTS = False
SHOW_GRADS = False
