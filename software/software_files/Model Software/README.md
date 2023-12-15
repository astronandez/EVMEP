# Model Creation Software File


## Model Development
For our purposes, we can leverage the diverse pool of computer vision model creation tools
publically available on github. The community version of Super-gradients is capable of exposing
a varienty of model parameters to the user, allowing fine grain model tuning for a variety of
machine learning and computer vision tasks. This library builds on top of pytorch and is compatible
with a majority of its librarys.

### Utilized Repositorys:
* Super-Gradients by Deci-AI (https://github.com/Deci-AI/super-gradients)
* Pytorch: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library [Conference paper]. Advances in Neural Information Processing Systems 32, 8024–8035. http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
* OpenCV-Python (https://github.com/opencv/opencv)
* TensorFlow: Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jozefowicz, R., Jia, Y., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Schuster, M., Monga, R., Moore, S., Murray, D., Olah, C., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., & Zheng, X. (2015). TensorFlow, Large-scale machine learning on heterogeneous systems [Computer software]. https://doi.org/10.5281/zenodo.4724125

### Public Training Datasets
COCO Dataset
* Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014). Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13 (pp. 740-755). Springer International Publishing.

#### Selected Model
Super-gradients recommends the small version of their flavor of You-Only-Look-Once (YOLO) Object Detection, for real time 
applications as its base latency is measured at approximatly 3.25 ms on an NVIDIA T4.

(insert image)

The motivation behind this baseline model selection was motivated by Super-gradients proprietary
Neural architecture search (NAS) algorithm used to automate the design of Artificial Neural Networks (ANN)
to help users more efficiently identify the optimal computer vision model configuration.

In particular, by taking advantange of the traditional YOLOv5 Backbone, an already fast implementation
of YOLO, is then further optimized on the pretrained weights of the COCO dataset. Then by using our available functions we can
leverage transfer learning to before training our own YOLO_NAS_S model on a custom dataset as described in the Dataset Software folder.

Finally, after training our model, we are left with roughly 19.02M parameters, far too big for resources limited platforms. Thus, to
circumvent this issue, we apply Quantization Aware Training (QAT) which is a form of quantization that converts FP32 values to UINT8,
effectively reducing our model file size to fit under more constricted resources

### Custom Configuration:
(insert Custom Super-Gradient Parameters)

### Final Performance Metrics:
(Insert Best Epoch Metrics)