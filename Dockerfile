# Dockerfile of Example
# Version 1.0
# Base Images
FROM registry.cn-shanghai.aliyuncs.com/aliseccompetition/tensorflow:1.4.0-devel-gpu
#MAINTAINER
MAINTAINER AlibabaSec

ADD . /competition

WORKDIR /competition
RUN pip install --upgrade pip
#RUN pip install --upgrade tensorflow-gpu==1.4.0
RUN pip --no-cache-dir install  -r requirements.txt
# INSTALL cleverhans foolbox

# pretrained_models
RUN mkdir ./models
RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/inception_v1.tar.gz' && tar -xvf inception_v1.tar.gz -C ./models/
RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/resnet_v1_50.tar.gz' && tar -xvf resnet_v1_50.tar.gz -C ./models/
RUN curl -O  'http://alg-misc.cn-hangzhou.oss.aliyun-inc.com/ijcai2019_ai_security_competition/pretrained_models/vgg_16.tar.gz' && tar -xvf vgg_16.tar.gz -C ./models/
