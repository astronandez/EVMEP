# Using the Arduino Nicla Vision for edge detection

## Description

*This is the code necesarry to setup and run our project on the edge using an arduino nicla vision.*

## Installation

### Prerequisites

Hardware Requirements: 
    1. Arduino Nicla Vision

Software Requirements:
    1. OpenMV
    2. Edge Impulse

### Setup

1. You will need to setup an Edge Impulse account and download the openMV IDE.
2. You will need to configure your software stack so that you cn compile code to the nicla vision. An example reference is here: https://docs.edgeimpulse.com/docs/development-platforms/officially-supported-mcu-targets/arduino-nicla-vision
3. Once you have setup the Nicla Vision with openMV and Edge Impulse, you need to train your dataset (if you our using ours then this step can be skipped for the most part - just download the folder named ucla-project-1-open-mv-fw-v1). A tutorial to do this can be found at the folowing link: https://www.hackster.io/mjrobot/tinyml-made-easy-object-detection-with-nicla-vision-407ddd
4. Now that you have succesfully trained and downloaded your model for object detection, we will deploy it to the Nicla Vision. First, connect the Nicla Vision with the openMV IDE once again.
5. When you try to connect the Nicla with the OpenMV IDE again, it will try to update its FW. Choose the option Load a specific firmware instead.
![Alt text](https://hackster.imgix.net/uploads/attachments/1638365/_ka3Y7z8V6I.blob?auto=compress%2Cformat&w=740&h=555&fit=max)
6. You will find a ZIP file on your computer from the Studio. Open it:
![Alt text](https://hackster.imgix.net/uploads/attachments/1638408/image_RWzBj7Cd5I.png?auto=compress%2Cformat&w=1280&h=960&fit=max)
7. After the download is finished, a pop-up message will be displayed. Press OK, and open the script ei_object_detection.py downloaded from the Studio.
8. Now press the green Play button to run the code.