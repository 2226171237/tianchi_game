# Dockerfile of Example
# Version 1.0
# Base Images
FROM registry.cn-shanghai.aliyuncs.com/aliseccompetition/tensorflow:1.1.0-devel-gpu
#MAINTAINER
MAINTAINER AlibabaSec

ADD . /competition

WORKDIR /competition
RUN pip install --upgrade pip
RUN pip install --upgrade tensorflow-gpu==1.4.0
RUN pip --no-cache-dir install  -r requirements.txt
# INSTALL cleverhans foolbox

# pretrained_models

RUN mkdir ./models
RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/inception_v1.tar.gz' && tar -xvf inception_v1.tar.gz -C ./models/
RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/resnet_v1_50.tar.gz' && tar -xvf resnet_v1_50.tar.gz -C ./models/
RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/vgg_16.tar.gz' && tar -xvf vgg_16.tar.gz -C ./models/

RUN CUDNN_DOWNLOAD_SUM=9b09110af48c9a4d7b6344eb4b3e344daa84987ed6177d5c44319732f3bb7f9c && \
    curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz -O && \
    echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v6.0.tgz" | sha256sum -c - && \
    tar --no-same-owner -xzf cudnn-8.0-linux-x64-v6.0.tgz -C /usr/local --wildcards 'cuda/lib64/libcudnn.so.*' && \
    rm cudnn-8.0-linux-x64-v6.0.tgz && \
    ldconfig
