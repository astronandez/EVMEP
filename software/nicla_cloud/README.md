# Using the Arduino Nicla Vision for cloud detection

## Description

*This is the code necesarry to setup and run our project on the cloud using an arduino nicla vision to stream video.*

## Installation

### Prerequisites

Hardware Requirements: 
    1. Arduino Nicla Vision

Software Requirements:
    1. OpenMV
    2. Cloud device/access

### Setup

1. You will need to setup an Edge Impulse account and download the openMV IDE.
2. You will need to configure your software stack (openMV) so that you can compile code to the nicla vision. An example reference is here: https://docs.arduino.cc/tutorials/nicla-vision/getting-started
3. Dowload the code in this repository.
4. Open the rtsp_video_server_wlan_1.py file and run the code on the Arduino Nicla Vision. This will begin an rtsp stream on your nicla vision that we can access from anywhere. Make sure you take note of your nicla visions ip address and port, we will need this information later.
5. On your cloud device or local machine open the nicla_cloud_detect_bb.py file.
6. Make necesarry updates such as updating the nicla vision ip address and port as well as changing your tensorflow model path.
7. Run the python file in your terminal.

*Port Forwarding: Depending on your setup you will need to enable port forwarding on your network. You will need to port forward the internal ip adress of your nicla vision to connect to the device. You will also need to port forward the rtsp stream port so that you can access it externally. Conatct your network provider for assistance doing this.*
