# Project Proposal

## 1. Motivation & Objective

Modern object-tracking methods are reliant on high-performance hardware due to the significant computational complexity required to produce a proper output. This complexity is introduced into the system as users seek to improve their model's detection accuracy [1]. A majority of embedded real-time intelligent traffic systems are focused on the position and detection of vehicle trajectories. However, due to limited hardware resources found on embedded platforms, these suffer in accurately considering the influence of the vehicle's acceleration and mass on the overall behavior [2]. We seek to implement, improve, and optimize a computer vision mass estimation pipeline based on the reference found in [2] on three distinct hardware platforms, each with two configurations: Edge and cloud computing services. From this cross-examination, our group will determine which implementation offers the optimal balance of cost and performance when comparing latency, accuracy, and system cost. 

## 2. State of the Art & Its Limitations

Current State-of-the-art computer-vision vehicle detection methods are capable of detecting vehicles and predicting trajectories using different object-tracking methods [3]. These systems are often referred to as Two-Stage Detectors, where the system analyzes the image in two stages. The first stage generates Regions of Interest (RoI), and the second stage uses those extracted features to detect the objects in each frame. In contrast, a Single-Stage detector instead detects and classifies the objects in one viewing of the image. When compared to each other, the two-stage detector method is able to produce higher object detection accuracy but at the cost of longer latency [5]. This is primarily due to the computational complexity of the two-stage detector, and as the number of objects of interest in the image increases, so too does the latency. As a result, these systems are challenging to implement for real-time embedded applications because of their reliance on high-performance computer hardware. However, recent developments in methods like You-Only-Look-Once (YOLO) offer a lightweight object-tracking solution that can be optimized to increase detection accuracy without drastically impacting latency, rivaling that of two-stage detectors.

## 3. Novelty & Rationale

We will be implementing an optimization of six new approaches. In previous work, this analysis was completed on a desktop computer without the assistance of an embedded system. We will be configuring six embedded systems with our reference pipeline and optimizing each to reduce latency and price. Additionally, we will be adding a new technique for increased mass estimation accuracy, which was previously not included in the pipeline. This approach will be successful because by testing the reference pipeline on different embedded platforms, we will identify the most cost-efficient platform. Additionally, by altering the original Kalman filter, our system will be capable of more accurately tracking vehicle trajectory because it will take into account additional error margins that can be used to estimate for mass with the least amount of error.

## 4. Potential Impact

If this project is successful, we will have found new ways to optimize a previously designed pipeline for vehicle mass estimation through computer vision but also have implemented the system on a new embedded platform that lowers both latency and cost while increasing efficiency. Additionally, we will develop a more accurate method for estimating vehicle mass by considering its physical properties. These developments can create impacts ranging from a more easily deployable intelligent traffic system for truck inspections to the capability to potentially identify the number of passengers in a vehicle (and, therefore, the vehicle's risk factor) all through a video stream. Additionally, we will have analyzed six different infrastructures for deploying computer vision object detection techniques on embedded frameworks for others to build on in the future.

## 5. Challenges

The most prominent challenge of this project is optimizing the aforementioned pipeline [2] to function on embedded systems on limited hardware while retaining detection and tracking accuracy. This challenge is magnified by our introduction of multiple software configurations across distinct embedded platforms. Each platform utilizes different hardware and will need to be optimized using strategies that are dedicated to each system. Our group will also be challenged with learning SOTA object detection methods to develop a deep understanding of the pipeline and areas of potential optimization, and learning how to harness cloud computing resources for our systems. A critical risk to our project is the potential of a non-functioning configuration. This may result from an embedded platform that is incapable of performing a complete cycle regardless of the inclusion of embedded optimization techniques.

## 6. Requirements for Success

The background knowledge necessary to perform the project will be centered around convolutional neural networks, Kalman Filtering, YOLO object tracking, and neural network optimization. In addition, our group will need to utilize embedded software optimization strategies like resource allocation, power consumption reduction, and code simplification to create a lightweight pipeline for our limited hardware. To develop our project, we will be utilizing the vehicle mass estimation pipeline proposed in [2] and a dataset containing images of vehicles of varying size and shape from a 3rd-person perspective. These images can be obtained from a publicly available dataset such as Open Images. 

## 7. Metrics of Success

For a successful project, our group will implement an object detection system on each platform and document the optimizations or reductions necessary to create each functioning program. 
After development, we will provide data-driven evidence supporting our claim of improved system performance when compared to the original pipeline configuration. To do so, we will utilize the metrics of latency, detection accuracy, and system cost and then graphically represent this data to draw conclusions on the most optimal configuration. We will also provide insight into our rationale for our optimizations and the sections of the pipeline that contributed to the largest performance hit on the systems.

## 8. Execution Plan

Our group will first source a dataset containing annotated images of various vehicles in several lighting and environmental conditions. Before training our model, we will modify our images to match the required inputs of our pipeline. We will then develop the pipeline outlined in the reference paper [2] to the best of our ability and document any differences that need to be made. Then, our group will optimize the reference pipeline on the Raspberry Pi 4 and document the optimizations and resources used to generate our Edge and cloud computing configurations. We will then repeat this process for the remaining platforms in the order of most to least capable. Lastly, we will summarize our optimizations and findings to determine the system that balances performance and cost.

## 9. Related Work

### 9.a. Papers

* DeepMaker: A multi-objective optimization framework for deep neural networks in embedded systems
    - https://doi.org/10.1016/j.micpro.2020.102989
    - This paper supports our claim that current computer vision systems are computationally complex and negatively impact the performance of embedded computer vision systems 
* Vehicle weight identification system for spatiotemporal load distribution on bridges based on non-contact machine vision technology and deep learning algorithms
    - https://doi.org/10.1016/j.measurement.2020.107801
    - This paper is the foundation of our project, as it provides our group with the pipeline we will use as our reference benchmark.
* A vision-based real-time traffic flow monitoring system for road intersections
    - https://doi.org/10.1007/s11042-023-14418-w
    - This paper provides evidence on the use of YOLO for object detection in real-time systems Computer Vision on real-time systems
* Embedded Sensors for Traffic Flow Monitoring
    - https://doi.org/10.1109/ITSC.2015.35
    - The paper provides motivation and use cases for embedded computer vision traffic systems
* Object detection using YOLO: challenges, architectural successors, datasets and applications
    - https://doi.org/10.1007/s11042-022-13644-y 
    - The paper provides an overview of YOLO and its performance against two-stage detectors

### 9.b. Datasets

* Open Images [6]
    - These images will contain the following categories: sedan, small truck, large truck, bus, school bus, and semi-truck.
    - The images will be at various angles and in different environments.


### 9.c. Software

* PyTorch v 2.1 [7]
* Arduino IDE v2.2.1 [8]
* OpenMV IDE v4.0.14 [9]
* OpenCV v4.8.0 [10]


## 10. References

[1]	M. Loni, S. Sinaei, A. Zoljodi, M. Daneshtalab, and M. Sjödin, “DeepMaker: A multi-objective optimization framework for deep neural networks in embedded systems,” Microprocessors and Microsystems, vol. 73, p. 102989, Mar. 2020, doi: https://doi.org/10.1016/j.micpro.2020.102989. 

[2]	Y. Zhou, Y. Pei, Z. Li, L. Fang, Y. Zhao, and W. Yi, “Vehicle weight identification system for spatiotemporal load distribution on bridges based on non-contact machine vision technology and deep learning algorithms,” Measurement, vol. 159, p. 107801, Jul. 2020, doi: https://doi.org/10.1016/j.measurement.2020.107801. 

[3]	Azimjonov, J., Özmen, A. & Varan, M. “A vision-based real-time traffic flow monitoring system for road intersections,” Multimed Tools Appl 82, 25155–25174 (2023). https://doi.org/10.1007/s11042-023-14418-w

[4]	M. Magrini, D. Moroni, G. Palazzese, G. Pieri, G. Leone and O. Salvetti, "Computer Vision on Embedded Sensors for Traffic Flow Monitoring," 2015 IEEE 18th International Conference on Intelligent Transportation Systems, Gran Canaria, Spain, 2015, pp. 161-166, https://doi.org/10.1109/ITSC.2015.35. 

[5]	Diwan, T., Anirudh, G. & Tembhurne, J.V. “Object detection using YOLO: challenges, architectural successors, datasets and applications,” Multimed Tools Appl 82, 9243–9275 (2023). https://doi.org/10.1007/s11042-022-13644-y 

[6]	A. Kuznetsova et al., “The Open Images Dataset V4,” International Journal of Computer Vision, Mar. 2020, doi: https://doi.org/10.1007/s11263-020-01316-z. 

[7]	A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” arXiv.org, 2019. https://arxiv.org/abs/1912.01703

[8]	Arduino IDE 2 (2022) Arduino S.r.l. [Online] https://www.arduino.cc/

[9]	OpenMV IDE (v4.0.14) OpenMV, LLC [Online] Accessed: Oct 28, 2023 https://openmv.io/ 

[10]	OpenCV (4.8.0) Open Source Computer Vision Library [Online] Accessed: Oct 28, 2023 https://opencv.org/ 

