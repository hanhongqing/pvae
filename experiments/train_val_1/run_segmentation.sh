#!/bin/bash
conda activate unsup_seg_vae
source ../config.txt
export CUDA_VISIBLE_DEVICES=3 && python ../../unsup_seg_vae/main_BetaSeg.py\
 --path_train $PATH_INPUT_BETASEG_HIGH_C1\
 --path_train $PATH_INPUT_BETASEG_HIGH_C2\
 --path_test $PATH_INPUT_BETASEG_HIGH_C3\
 --path_output $PATH_OUTPUT\
 --path_model $PATH_MODEL\
 --path_representation $PATH_REPRESENTATION\
 --path_loss_plot $PATH_LOSS_PLOT\
 --path_loss_text $PATH_LOSS_TEXT\
 --path_posterior_text $PATH_POSTERIOR_TEXT\
 --path_posterior_plot $PATH_POSTERIOR_PLOT\
 --path_suffix_mask_cell $SUFFIX_CELL\
 --path_suffix_mask_component $SUFFIX1\
 --path_suffix_mask_component $SUFFIX2\
 --path_suffix_mask_component $SUFFIX3\
 --path_suffix_mask_component $SUFFIX4\
 --path_suffix_mask_component $SUFFIX5\
 --path_suffix_mask_component $SUFFIX6\
 --path_suffix_mask_component $SUFFIX7\
 --max_intensity 256\
 --num_workers_test 32\
 --n_train $N_TRAIN_CROSSVAL\
 --n_epoch 1\
 --run_train\
 --run_train_visualize\
 --run_infer\
 --run_cluster\


conda deactivate

