#!/bin/bash

sed -e 's/{image}/tensorflow\/tensorflow:latest-py3/g' Dockerfile_template > Dockerfile
uid=$(id $whoamai | awk '{print $1}' | sed -e 's/uid=//g' | sed -e 's/(.*$//g')
gid=$(id $whoamai | awk '{print $2}' | sed -e 's/gid=//g' | sed -e 's/(.*$//g')
sudo docker build --build-arg uid=$uid --build-arg gid=$gid -t takuseno/beta-vae .
