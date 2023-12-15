# Using the Raspberry pi4 for cloud detection

## Description

*This is the code necesarry to setup and run our project on the cloud using a Raspberry pi4 (bullseye) to stream video.*

## Installation

### Prerequisites

Hardware Requirements: 
    1. Raspbery pi4
    2. Your choice of camera -> We chose to use the Raspberry pi camera module v2

Software Requirements:
    1. python3
    2. Cloud device/access

### Setup

1. You will need to setup your raspberry pi and configure your local camera, general instructions for this can be found at the following link: https://raspberrytips.com/install-camera-raspberry-pi/
2. in general it is very useful to setup ssh for your raspberry pi, a general guide can be found here: https://phoenixnap.com/kb/enable-ssh-raspberry-pi
3. Dowload the code in this repository to both your raspberry pi and your cloud/local machine.
4. On your raspberry pi we will use the v4l2rtspserver and ffmeg setup the rtsp stream (udp). Run the following command in your terminal: 'v4l2rtspserver -F 10 -H 240 -W 320 -P 8554 &' to start your rtsp stream on port 8554. You may need to install dependencies on your device, install whatever dependencies are asked for.
5. On your cloud device or local machine open the pi_cloud_detect_bb.py file.
6. Make necesarry updates such as updating the nicla vision ip address and port as well as changing your tensorflow model path.
7. Run the python file in your terminal.

*Port Forwarding: Depending on your setup you will need to enable port forwarding on your network. You will need to port forward the internal ip adress of your nicla vision to connect to the device. You will also need to port forward the rtsp stream port so that you can access it externally. Conatct your network provider for assistance doing this.*
