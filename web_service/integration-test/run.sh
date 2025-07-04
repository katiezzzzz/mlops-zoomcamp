#!/usr/bin/env bash

# exit at first error seen
# set -e

# cd to the directory of this script
cd "$(dirname "$0")"

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="web-service:${LOCAL_TAG}"
    echo "No local image name provided, building a new image with tag ${LOCAL_TAG}."
    docker build -t ${LOCAL_IMAGE_NAME} ..
else
    echo "Using local image: ${LOCAL_IMAGE_NAME}"
fi

docker-compose up -d

sleep 1

python test_docker.py

ERROR_CODE=$?

if [ $ERROR_CODE -ne 0 ]; then
    docker-compose logs
    echo "Integration test failed with error code: $ERROR_CODE"
fi

docker-compose down
exit $ERROR_CODE
