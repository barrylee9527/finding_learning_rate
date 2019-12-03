import os
CLASSES = ["ballooning","normal","steatosis"]
MIN_LR = 1e-4
MAX_LR = 1e-1
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 60
LRFIND_PLOT_PATH = os.path.sep.join(["/cptjack/totem/barrylee/codes/output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["/cptjack/totem/barrylee/codes/output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["/cptjack/totem/barrylee/codes/output", "clr_plot.png"])

