#!/bin/bash

# Exit if any cmd fails
set -e

if (( $EUID == 0 )); then
    echo "Please add yourself to the docker group and run script not as root"
    exit 1
fi

# Location of this script
BUILD_CONTEXT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

xhost +local:docker
docker build \
  --build-arg USERNAME=$USER \
  --build-arg UID=$EUID \
  -t kimera-evaluation:master \
  -f ./Dockerfile ..
xhost -local:docker
