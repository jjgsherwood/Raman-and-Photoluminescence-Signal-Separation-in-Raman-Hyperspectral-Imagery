NUM_WORKERS = 0
VALIDATION_PER = 0.1

# determine after how many epochs the training phases switch.
PRETRAINING_PHASE_1 = 3
PRETRAINING_PHASE_2 = 6

NN_INPUTS = {
    'cuda': True,
    'log_step': 5000,
    'lr': 0.001
}

# show extra plots
SHOW_WEIGHTS = False
SHOW_GRADS = False
