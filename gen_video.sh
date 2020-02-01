#!/bin/sh

ffmpeg -r 30 -f image2 -i $1_%04d.png -vcodec libx264 -crf 18 -pix_fmt yuv420p -vf scale='trunc(iw/2)*2:trunc(ih/2)*2' $1.mp4
