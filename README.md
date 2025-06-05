# mlops-zoomcamp

### Build image from Dockerfile
```
docker build -t web-service:v3 .
```

### For running docker image from web_service and expose port to host machine
```
docker run -it --rm \
    -p 9696:9696 \
    -e MODEL_LOCATION="/app/model" \
    -v $(pwd)/model:/app/model \
    web-service:v3
```