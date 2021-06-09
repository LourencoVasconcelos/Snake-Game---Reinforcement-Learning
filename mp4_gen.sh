#!/bin/bash

FNAME=${1:-%04d.png}
ONAME=${2:-out.mp4}
FPS=${3:-25}
echo ffmpeg -framerate $FPS -i $FNAME  -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=$FPS" -pix_fmt yuv420p $ONAME