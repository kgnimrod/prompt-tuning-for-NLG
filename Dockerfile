FROM nvidia/cuda:11.2.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends -y python3.8 python3-pip python3.8-dev python3.8-venv build-essential git
RUN python3 -m pip install -U pip && \
    pip install --upgrade pip setuptools

RUN mkdir -p /opt/workspace/
WORKDIR /opt/workspace/

COPY ./requirements ./requirements

RUN pip install -r requirements

CMD ./run.sh
# CMD bash