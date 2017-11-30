#!/bin/sh
python3 2D_SPH/scenes/tools/train_sequential.py var 1 epochs 250 data_start 0 val_split 0.2 time_start 5 ref 2D_data/patches/highres/ref_sph_2D_v02-01_d%03d_var%02d_%03d batch 32 data_end 18 log_intervall 10 fig models/2D/sph_2D_v10_loss model models/2D/sph_2D_v10 checkpoint_intervall 10 time_end 15 src 2D_data/patches/lowres/sph_2D_v02-01_d%03d_var%02d_%03d features ps
