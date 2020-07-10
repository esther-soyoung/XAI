FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python3-pip python3-wheel python3-setuptools gcc g++
RUN apt-get install -y git

RUN python3.6 -m pip install pip --upgrade

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

ADD . /XAI

WORKDIR /XAI

RUN pip3 install -r requirements.txt

EXPOSE 80

CMD python app.py