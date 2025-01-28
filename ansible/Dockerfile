FROM python:3.11-alpine

RUN apk add --no-cache bash openssh-client sshpass docker-cli python3 py3-pip
RUN pip install ansible docker requests

WORKDIR /ansible
COPY ansible/ .