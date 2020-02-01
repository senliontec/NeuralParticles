#!/bin/sh
for i in 0 1 2 3 4
do
    python -m neuralparticles.tools.show_detail_csv src 2D_data/tmp/2018-08-27_14-13-57/eval/eval_patch_src_e000_d00${i}_t000.csv res 2D_data/tmp/2018-08-27_14-13-57/eval/eval_patch_res_e002_d00${i}_t000.csv idx $1 out details_d/detail_i%04d_%s_${i}
done