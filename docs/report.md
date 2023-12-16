# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Intelligent traffic systems (ITS) are becoming more common as cities continue to trend toward smart infrastructure. Sensors and systems like LIDAR, Radar, and Induction coils contribute toward smarter infrastructure but are limited to information a smaller subset of information and face scalability issues and high installation costs. In contrast, vision-based traffic solutions offer an in-depth analysis of the environment that can improve traffic management. Current, real-time computer-vision traffic systems have been capable of detecting vehicles and predicting trajectories but are computationally demanding and challenging to implement on resource-limited platforms. Nevertheless, advancements in Edge platforms and lightweight computer-vision object-tracking strategies make embedded systems an affordable and scalable alternative to larger systems. We’ll focus on optimizing our referenced pipeline on three distinct embedded platforms: Arduino® Nano 33 IoT, Arduino® Nicla Vision, and the Raspberry Pi 4, each with two configurations Edge and cloud computing. With the information gathered from our evaluation, our group will be able to determine the system that provides the best balance of cost and performance.

# 1. Introduction

This section should cover the following items:

* Motivation & Objective: What are you trying to do and why? (plain English without jargon)
* State of the Art & Its Limitations: How is it done today, and what are the limits of current practice?
* Novelty & Rationale: What is new in your approach and why do you think it will be successful?
* Potential Impact: If the project is successful, what difference will it make, both technically and broadly?
* Challenges: What are the challenges and risks?
* Requirements for Success: What skills and resources are necessary to perform the project?
* Metrics of Success: What are metrics by which you would check for success?

# 2. Related Work

# 3. Technical Approach
### Dataset Technical Approach:
Dataset choice considerations:
We will focus on Single-Object Detection as our hardware platforms are heavily resource constrained and processing multiple detections will negatively impact our real-time performance on both the edge and cloud. Images should be of a single vehicle
The dataset must be large spanning several distinct vehicle body style categories
The dataset must contain several viewing angles of vehicles
We must be able to derive a Curb Weight value from the available information
For our project our group used the following publicly available datasets:
* CompCars
  * This dataset contains 214,345 images of 1,687 car models from both web and surveillance-sources
  * This dataset was hierarchy organized in a folder for both images and labels: /{User_Root_Directory}/images/make_id/model_id/imageexample.jpg
.Txt files provided by author outlined car_type_id with small modifications: Car Types: ['MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 'wagon', 'pickup', 'hardtop convertible', 'sports', 'crossover', 'convertible']. Where the first entry was labeled int(1) and 0/0.0 entries are unknown or uncategorized, this category was dramatically overrepresented and thus only the images with categorized car_type_id were retained leaving 76394 images and labels
  * An additional ‘attributes.txt’ file is included relating model_id to different vehicle features in the following: [‘model_id, maximum_speed, displacement, door_number, seat_number, car_type'].

We selected this dataset primarily because of the large extent of represented vehicles at several distinct viewing angles. Additionally, each vehicle image was linked to individual text files with information about the Car Type and bounding box. 
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. A Large-Scale Car Dataset for Fine-Grained Categorization and Verification, In Computer Vision and Pattern Recognition (CVPR), 2015. Link: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html 

* Carsheet.io
  * https://carsheet.io/ (Acquired dataset using a web scraping tool with permission from the original author given through email on Nov 16th)

### Model Technical Analysis:
For our purposes, we can leverage the diverse pool of computer vision model creation tools
publically available on github. The community version of Super-gradients is capable of exposing
a varienty of model parameters to the user, allowing fine grain model tuning for a variety of
machine learning and computer vision tasks. This library builds on top of pytorch and is compatible
with a majority of its librarys.

Super-gradients recommends the small version of their flavor of You-Only-Look-Once (YOLO) Object Detection, for real time 
applications as its base latency is measured at approximatly 3.25 ms on an NVIDIA T4.

The motivation behind this baseline model selection was motivated by Super-gradients proprietary
Neural architecture search (NAS) algorithm used to automate the design of Artificial Neural Networks (ANN)
to help users more efficiently identify the optimal computer vision model configuration.

In particular, by taking advantange of the traditional YOLOv5 Backbone, an already fast implementation
of YOLO, is then further optimized on the pretrained weights of the COCO dataset. Then by using our available functions we can
leverage transfer learning to before training our own YOLO_NAS_S model on a custom dataset as described in the Dataset Software folder.

Finally, after training our model, we are left with roughly 19.02M parameters, far too big for resources limited platforms. Thus, to
circumvent this issue, we apply Quantization Aware Training (QAT) which is a form of quantization that converts FP32 values to UINT8,
effectively reducing our model file size to fit under more constricted resources

# 4. Evaluation and Results



# 5. Discussion and Conclusions



# 6. References
[1]	M. Loni, S. Sinaei, A. Zoljodi, M. Daneshtalab, and M. Sjödin, “DeepMaker: A multi-objective optimization framework for deep neural networks in embedded systems,” Microprocessors and Microsystems, vol. 73, p. 102989, Mar. 2020, doi: https://doi.org/10.1016/j.micpro.2020.102989.

[2]	Y. Zhou, Y. Pei, Z. Li, L. Fang, Y. Zhao, and W. Yi, “Vehicle weight identification system for spatiotemporal load distribution on bridges based on non-contact machine vision technology and deep learning algorithms,” Measurement, vol. 159, p. 107801, Jul. 2020, doi: https://doi.org/10.1016/j.measurement.2020.107801. 

[3]	Azimjonov, J., Özmen, A. & Varan, M. “A vision-based real-time traffic flow monitoring system for road intersections,” Multimed Tools Appl 82, 25155–25174 (2023). https://doi.org/10.1007/s11042-023-14418-w

[4]	M. Magrini, D. Moroni, G. Palazzese, G. Pieri, G. Leone and O. Salvetti, "Computer Vision on Embedded Sensors for Traffic Flow Monitoring," 2015 IEEE 18th International Conference on Intelligent Transportation Systems, Gran Canaria, Spain, 2015, pp. 161-166, https://doi.org/10.1109/ITSC.2015.35. 

[5]	Diwan, T., Anirudh, G. & Tembhurne, J.V. “Object detection using YOLO: challenges, architectural successors, datasets and applications,” Multimed Tools Appl 82, 9243–9275 (2023). https://doi.org/10.1007/s11042-022-13644-y 

[6]	A. Kuznetsova et al., “The Open Images Dataset V4,” International Journal of Computer Vision, Mar. 2020, doi: https://doi.org/10.1007/s11263-020-01316-z. 

[7]	A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” arXiv.org, 2019. https://arxiv.org/abs/1912.01703

[8]	Arduino IDE 2 (2022) Arduino S.r.l. [Online] https://www.arduino.cc/

[9]	OpenMV IDE (v4.0.14) OpenMV, LLC [Online] Accessed: Oct 28, 2023 https://openmv.io/ 

[10] OpenCV (4.8.0) Open Source Computer Vision Library [Online] Accessed: Oct 28, 2023 https://opencv.org/

[11] COCO Dataset: Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13 (pp. 740-755). Springer International Publishing.

[12] TensorFlow: Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jozefowicz, R., Jia, Y., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Schuster, M., Monga, R., Moore, S., Murray, D., Olah, C., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., & Zheng, X. (2015). TensorFlow, Large-scale machine learning on heterogeneous systems [Computer software]. https://doi.org/10.5281/zenodo.4724125

[13] Super-Gradients by Deci-AI (https://github.com/Deci-AI/super-gradients)

[14]Pytorch: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019).
