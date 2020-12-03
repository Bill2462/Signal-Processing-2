# use debian buster as a base image
FROM debian:buster

# create project directory and copy all the files
RUN mkdir /usr/src/report && mkdir /usr/src/report/build
WORKDIR /usr/src/report

# install dependencies
RUN apt update && apt install -y python3-pip pandoc texlive-xetex python3-opencv libopencv-dev ffmpeg

# build
COPY requirements.txt .

# install jupyter notebook
RUN pip3 install -r requirements.txt

RUN rm requirements.txt

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /usr/src/report
USER user

# launch jupyter on startup
CMD jupyter lab --ip=0.0.0.0
