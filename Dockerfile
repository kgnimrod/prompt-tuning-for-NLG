FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3.8 python3-pip python3.8-dev python3.8-venv build-essential git
RUN python3.8 -m pip install -U pip && \
    pip3 install --upgrade pip setuptools

RUN mkdir -p /opt/workspace/
WORKDIR /opt/workspace/

Run ./requirements ./requirements

RUN pip3 install -r requirements

CMD ./run.sh