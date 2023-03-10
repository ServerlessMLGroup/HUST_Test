docker build . -t multi_model_pytorch
docker run --name multi_model_pytorch -d -it --gpus all --runtime nvidia -v ~/HUST_Test:/workspace/HUST_Test ssd_pytorch:latest /bin/bash
docker exec -ti multi_model_pytorch /bin/bash
# docker stop multi_model_pytorch
# docker rm multi_model_pytorch