ARG FROM_IMAGE_NAME=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM ${FROM_IMAGE_NAME}
# Set working directory
WORKDIR /workspace

# Copy the model files
COPY . .

RUN apt-get -y upgrade
RUN apt-get -y update
RUN apt-get -y install git

# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

ENV CUDNN_V8_API_ENABLED=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
