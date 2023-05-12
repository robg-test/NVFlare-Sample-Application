FROM python:3.8.10

RUN apt-get update && apt-get install -y dos2unix

RUN pip install -U pip
COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

RUN python3 -c "import torchvision;torchvision.datasets.CIFAR10(root='/root/data/', download=True)"

WORKDIR /nvflare

COPY ./app /nvflare/app/
COPY ./utils/test.py /nvflare/
COPY ./utils/start_nvflare_simulator.sh /nvflare/

# Fix the Docker issue with line endings from .sh files when running it on Windows
# from https://github.com/docker/for-win/issues/1340
RUN dos2unix /nvflare/start_nvflare_simulator.sh
RUN chmod 777 /nvflare/start_nvflare_simulator.sh

ENV PATH="${PATH}:/nvflare/"
