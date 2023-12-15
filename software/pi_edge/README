# Using the Raspberry pi4 for edge detection

## Description

*This is the code necesarry to setup and run our project on the edge using a Raspberry pi4 (bullseye).*

## Installation

### Prerequisites

Hardware Requirements: 
    1. Raspbery pi4
    2. Your choice of camera -> We chose to use the Raspberry pi camera module v2

Software Requirements:
    1. python3

### Setup

1. You will need to setup your raspberry pi and configure your local camera, general instructions for this can be found at the following link: https://raspberrytips.com/install-camera-raspberry-pi/
2. in general it is very useful to setup ssh for your raspberry pi, a general guide can be found here: https://phoenixnap.com/kb/enable-ssh-raspberry-pi
3. Dowload the code in this repository to your raspberry pi.
4. On your raspberry pi we will use the v4l2rtspserver and ffmeg setup the rtsp stream (udp). Run the following command in your terminal: 'v4l2rtspserver -F 10 -H 240 -W 320 -P 8554 &' to start your rtsp stream on port 8554. You may need to install dependencies on your device, install whatever dependencies are asked for.
5. On your raspberry pi open the local.py file.
6. Make necesarry updates such as updating your tensorflow model path.
7. Run the python file in your terminal.
