#!/bin/bash
# Test Random Forest training using existing generic SLURM template

./scripts/run_train_tml.sh random_forest sup ./models/test_rf/ --trials 3 --cv 2 --feature-selection SelectKBest
