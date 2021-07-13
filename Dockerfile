############Dockerfile###########
FROM openjdk:8

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y git 
RUN apt-get install -y curl
RUN apt-get install -y vim
RUN apt-get install -y tar
RUN apt-get install -y bzip2

RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip

####
RUN pip3 install gdown==3.12.2
RUN pip3 install rdflib==5.0.0
RUN pip3 install requests==2.24.0
RUN pip3 install pandas==1.1.3
RUN pip3 install elasticsearch==7.11.0
RUN pip3 install pyspark==3.1.1
RUN pip3 install esdk-obs-python==3.20.11 --trusted-host pypi.org
RUN pip3 install Pillow==8.2.0
RUN pip3 install xlrd==1.1.0
RUN pip3 install xlsxwriter==1.4.3
RUN pip3 install sklearn==0.0

####

RUN pip3 install h5py==2.10.0
RUN pip3 install tensorflow==1.14.0
RUN pip3 install keras==2.2.4

####

WORKDIR /

####

ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

####

WORKDIR /
RUN wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_amd64.tar.gz
RUN tar -xzvf obsutil_linux_amd64.tar.gz

WORKDIR /

RUN echo "1sd6g1s6g15"
############Dockerfile############
