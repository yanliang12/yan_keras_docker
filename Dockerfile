############Dockerfile###########
FROM openjdk:8u292-jre

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
RUN pip3 install h5py==2.10.0
RUN pip3 install tensorflow==1.14.0
RUN pip3 install keras==2.2.4

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

####
RUN pip3 install matplotlib==3.4.2
RUN pip3 install scikit-learn==0.24.2
RUN pip3 install pandasql==0.7.3

WORKDIR /

####
RUN apt-get install -y python3 
RUN apt-get install -y python3-dev 
RUN apt-get install -y python3-pip

RUN apt-get install -y python
RUN apt-get install -y python-dev 
RUN apt-get install -y python-pip

RUN apt-get install -y gcc 
RUN apt-get install -y libssl-dev 
RUN apt-get install -y libffi-dev 
RUN apt-get install -y libxml2-dev 
RUN apt-get install -y libxslt1-dev 
RUN apt-get install -y zlib1g-dev 
RUN apt-get install -y build-essential 

RUN pip3 install jupyterlab==3.0.16
RUN pip3 install notebook==6.4.0

####

ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3

####

RUN echo "1sd6g1s6g15"

WORKDIR /

CMD jupyter notebook --ip 0.0.0.0 --port 9971 --NotebookApp.token='' --no-browser --allow-root 
############Dockerfile############
