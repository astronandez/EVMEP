# Table of Contents
* Abstract
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach) 
* [Evaluation and Results](#4-evaluation-and-results) 
* [Discussion and Conclusions](#5-discussion-and-conclusions) 
* [References](#6-references) X

# Abstract

Intelligent traffic systems (ITS) are becoming more common as cities continue to trend toward smart infrastructure. Sensors and systems like LIDAR, Radar, and Induction coils contribute toward smarter infrastructure but are limited to a smaller subset of information and face scalability issues and high installation costs. In contrast, vision-based traffic solutions offer an in-depth analysis of the environment that can improve traffic management. Current, real-time computer-vision traffic systems have been capable of detecting vehicles and predicting trajectories but are computationally demanding and challenging to implement on resource-limited platforms. Nevertheless, advancements in Edge platforms and lightweight computer-vision object-tracking strategies make embedded systems an affordable and scalable alternative to larger systems. We’ll focus on optimizing our referenced pipeline on three distinct embedded platforms: Arduino® Nicla Vision, and the Raspberry Pi 4, each with two configurations Edge and cloud computing. With the information gathered from our evaluation, our group will be able to determine the system that provides the best balance of cost and performance.

# 1. Introduction

Advancements in computer vision technology have propelled the feasibility of real-time computer vision applications. We seek to emulate a Vehicle detection pipeline that was configured for high-performance hardware on resource-constrained platforms of Raspberry Pi 4 and the Nicla Vision PCB. By bringing such applications to the edge, we can create unique and sophisticated solutions to problems involving object, image, and gesture recognition. Today, most computer vision models are tuned to function on systems with high-performance components like a dedicated GPU or multi-core processors. These systems leverage a convolutional neural network architecture to extract features from images and use those features, such as lines, color, and edges, to determine what an object is. However, these computations are difficult to process with limited resources, and typically introduce latency into real-time applications. In our approach, we take advantage of new single-shot-detector models that have a compelling performance-to-latency ratio for embedded systems. When compared to traditional models, these SSDs can perform near or at times better than their more robust counterparts of Two-stage-detectors like VGG-16. If successful, we will have identified impactful computer vision model optimization methods and viable strategies for future deployment on the edge for other analogous computer vision projects. The background knowledge necessary to perform the project will be centered around convolutional neural networks, Kalman Filtering, YOLO object tracking, and neural network optimization. In addition, our group will need to utilize embedded software optimization strategies like resource allocation, power consumption reduction, and code simplification to create a lightweight pipeline for our limited hardware. The most critical metrics for our project are latency, hardware/cloud implementation cost, and model accuracy. These three metrics together help provide our group with a decision on which model is going to be best for our edge application.

# 2. Related Work

Current State-of-the-art computer-vision vehicle detection methods are capable of detecting vehicles and predicting trajectories using different object-tracking methods [3]. These systems are often referred to as Two-Stage Detectors, where the system analyzes the image in two stages. The first stage generates Regions of Interest (RoI), and the second stage uses those extracted features to detect the objects in each frame. In contrast, a Single-Stage detector instead detects and classifies the objects in one viewing of the image. When compared to each other, the two-stage detector method is able to produce higher object detection accuracy but at the cost of longer latency [5]. This is primarily due to the computational complexity of the two-stage detector, and as the number of objects of interest in the image increases, so too does the latency. As a result, these systems are challenging to implement for real-time embedded applications because of their reliance on high-performance computer hardware. However, recent developments in methods like You-Only-Look-Once (YOLO) offer a lightweight object-tracking solution that can be optimized to increase detection accuracy without drastically impacting latency, rivaling that of two-stage detectors. We aim to implement these new model training methodologies on embedded platforms to identify vehicle mass. Given a pipeline for vehicle mass estimation that uses high accuracy equipment but is not in real-time, can we implement the same tasks and how efficient can our systems be in terms of cost, accuracy, and latency?

# 3. Technical Approach
### Dataset Technical Approach:
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
 * This dataset was used to create a correlation between our vehicle types and their curb weight as defined by the manufacturer. Curb weight refers to the weight of the vehicle without any additional load, including passengers. 

Additionally, for one of our methods (nicla edge detection) we had to train a custom model using not “YOLO” but “FOMO” - Faster Objects More Objects. FOMO is a more recent development and is designed to be a simpler and faster object detection model compared to traditional methods like SSD or YOLO. The primary focus of FOMO is to achieve rapid object detection while maintaining high accuracy, especially in scenarios where there are many objects to detect in an image. It simplifies the architecture and computational process compared to more complex models, which makes it efficient for real-time applications and devices with limited computational resources.

We utilized openMV and Edge Impulse to take our custom YOLO dataset and train it to mee the necessary memory requirements to complete object detection on the edge using the nicla vision (model needs to be ~766 KB). This led to a much more inaccurate model, seen in the F1 score below for the trained dataset.


![FOMO F1 Score](EVMEP/data/Edge Impulse metrics/f1_score.jpg)

### Hardware/Software Technical Approach:
For our project we had the task of making four implementations of an embedded mass estimation pipeline. 

# 4. Evaluation and Results

## Dataset
The following displays the final configuration of our dataset contained in the final_data_config.csv file
in our data folder:

|car_type_id|x_1               |y_1               |x_2               |y_2               |year|make_name       |model_name               |car_type_name      |make_id|model_id|avg_curb_weight|
|-----------|------------------|------------------|------------------|------------------|----|----------------|-------------------------|-------------------|-------|--------|---------------|
|0          |0.5022026431718062|0.5678913738019169|0.7951541850220264|0.8642172523961661|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.5099009900990099|0.5669856459330144|0.5291529152915292|0.6140350877192983|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.5198237885462555|0.5638977635782748|0.8303964757709251|0.8083067092651757|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.4906387665198238|0.5990415335463258|0.8623348017621145|0.7603833865814696|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.4900881057268722|0.5862619808306709|0.7687224669603524|0.5239616613418531|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |

We verified the accuracy of our dataset by querying two publicly available datasets in the following fashion:
  * In the original data, 0 was an unclassified vehicle in the Car Types classes, this
  classification was removed to prevent overrepresentation of a type in the dataset
  * Estate Car Type was converted to Wagon
  * Reformatted bounding box coordinates into Yolo format from Pascal VOC
    * x1y1x2y2 -> center x center y width height (normalized 0 to 1)
  * Compiled all relevent data into a single and easily indexible CSV rather than
  having multiple text file interfaces to access information
    * Original data config stored car attributes like Model, Make, etc. in a seperate .txt file from
  that would not function with the standard dataset configuration for our model

Then to correlate our CompCars data to our webscraped data we performed the following modifications to the dataset:
 * Query both the CompCars and Carsheets.io dataset to find matching Make and Model pairs
  * If their is an identified pair, directly transfer the weight from carsheets to CompCars
  * Else create a seperate dataframe containing all uncompatible models
 * Query all defined individual curb weights and group them by CompCars Car Type
 * Remove duplicate entries resulting from identical models by weight over years
 * Find the average weight for the remaning classified Car Types
 * Used the averge weight for each Car Type to fill missing information in our avg_curb_weight column.


## Model

From our investigations, SSDs were the most viable options for embedded platforms; in particular, our group investigated the YOLO NAS S version of the YOLO architecture by Deci Ai. The motivation behind this baseline model selection was motivated by Super-gradients proprietary
Neural architecture search (NAS) algorithm used to automate the design of Artificial Neural Networks (ANN). Additionally, the Super-gradients repository contains features that allow users 
users more efficiently identify the optimal computer vision model configuration by exposing
a varienty of model parameters to the user. This allows for fine-grain model tuning for a variety of machine learning and computer vision tasks. This library builds on top of Pytorch and is compatible with a majority of its librarys.

<img src="https://github.com/Deci-AI/super-gradients/raw/master/documentation/source/images/yolo_nas_frontier.png" height="600px">

Super-gradients recommends the small version of their flavor of You-Only-Look-Once (YOLO) Object Detection, for real time 
applications as its base latency is measured at approximatly 3.25 ms on an NVIDIA T4.

In particular, by taking advantange of the traditional YOLOv5 Backbone, an already fast implementation
of YOLO, is then further optimized on the pretrained weights of the COCO dataset. Then by using our available functions we can
leverage transfer learning to before training our own YOLO_NAS_S model on a custom dataset as described in the Dataset Software folder.

Finally, after training our model, we are left with roughly 19.02M parameters, far too big for resources limited platforms. Thus, to circumvent this issue, we apply Quantization Aware Training (QAT) which is a form of quantization that converts FP32 values to UINT8, effectively reducing our model file size to fit under more constricted resources


# 5. Discussion and Conclusions

### Dataset and Model
From our investigation, we’ve noticed several important factors that will influence the performance and usability of the dataset:
* Platforms like Roboflow enable fast and quick conversion of data elements like bounding box coordinates that are compatible with a user-defined model. The dataset made and used by our group can be found here: https://universe.roboflow.com/research-projects-qodgb/vehicle-body-style-dataset
* When using web-scraped images of only 1 object in the image, it will be more difficult to detect said object when there are multiple classifiable objects in the computer vision view.
* Data variety is crucial to its performance; ensuring that there is an even distribution of classes across training, validation, and testing is a must. This will help ensure that there are no overrepresented groups in the dataset that would skew prediction results.
* When selecting a model, image size, quantity, and resolution will influence its overall size and can be downsized by using quantization techniques and traditional neural network pruning.
* Even YOLO, at its most miniature format, can still prove to be too large for some platforms like the Nicla Vision regardless of the optimization steps used to reduce the overall model size.


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
