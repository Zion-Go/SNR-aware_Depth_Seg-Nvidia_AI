#!/bin/bash

cd "$(dirname "$0")"
cd ..
export PRODUCTION_PATH=$PWD
export ARCH=`uname -m`
export NUM_THREADS=`nproc`

xhost +
docker-compose --env-file $PRODUCTION_PATH/up.env \
    -f $PRODUCTION_PATH/up.yml \
    up $@
xhost -
