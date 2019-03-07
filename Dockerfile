FROM nvidia/cuda:8.0-cudnn7-devel-centos7
LABEL maintainer "bbp94"
 
# 安装bzip2
RUN yum -y install bzip2
 
# 安装ffmpeg
RUN yum install -y epel-release && \
    rpm --import /etc/pki/rpm-gpg/RPM-GPG-KEY-EPEL-7 && \
    rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro && \
    rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm && \
    yum install -y ffmpeg
 
# 安装Anaconda
COPY ./Anaconda3-4.2.0-Linux-x86_64.sh /tmp/Anaconda3-4.2.0-Linux-x86_64.sh
WORKDIR /tmp
RUN sh -c '/bin/echo -e "\nyes\n\nyes" | sh Anaconda3-4.2.0-Linux-x86_64.sh'
 
#设置环境变量
ENV PATH /root/anaconda3/bin:$PATH
 
#安装opencv 和 threadpool 和 pytorch
COPY ./opencv_python-3.4.3.18-cp35-cp35m-manylinux1_x86_64.whl /tmp/opencv_python-3.4.3.18-cp35-cp35m-manylinux1_x86_64.whl
COPY ./torch-0.4.1-cp35-cp35m-linux_x86_64.whl /tmp/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
 
RUN pip install threadpool && \
    pip install opencv_python-3.4.3.18-cp35-cp35m-manylinux1_x86_64.whl && \
    pip install torch-0.4.1-cp35-cp35m-linux_x86_64.whl && \
    pip install torchvision
 
# 设置软连接
RUN rm -rf /usr/bin/python && ln -s /root/anaconda3/bin/python /usr/bin/python