#!/bin/bash
# Test DNN5 training using existing generic SLURM template
# Uses max_epochs=1500 with early_stopping_patience=100

./scripts/run_train_dl.sh deepdense reh ./models/test_dnn5/ --trials 3 --cv 2
