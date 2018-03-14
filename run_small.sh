#!/bin/sh

python3 gen_src.py config $1 data $2 manta $3
python3 train.py config $1 data $2 manta $3
python3 run.py config $1 data $2
python3 show_data.py config $1 data $2 manta $3 type "res"