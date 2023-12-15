# Dataset and Model create Software File


## Dataset Development
We can train a computer vision model by compiling a large data 
pool that relates car different car models to an averaged car type weight. 
For our data set we've classifed cars into the following categories:

Car Types: 'MPV', 'SUV', 'sedan', 'hatchback', 'minibus', 'fastback', 'wagon',
'pickup', 'hardtop convertible', 'sports', 'crossover', 'convertible'

Number of Car Type = 12

### Utilized Datasets:
* CompCars
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. A Large-Scale Car Dataset for Fine-Grained Categorization and Verification, In Computer Vision and Pattern Recognition (CVPR), 2015. Link: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html

* Carssheet.io [Publically available, permission granted from author]

### Dataset Modifications:
* CompCars
  * In the original data, 0 was an unclassified vehicle in the Car Types classes, this
  classification was removed to prevent overrepresentation of a type in the dataset
  * Estate Car Type was converted to Wagon
  * Reformatted bounding box coordinates into Yolo format from Pascal VOC
    * x1y1x2y2 -> center x center y width height (normalized 0 to 1)
  * Compiled all relevent data into a single and easily indexible CSV rather than
  having multiple text file interfaces to access information
    * Original data config stored car attributes like Model, Make, etc. in a seperate .txt file from
  that would not function with the standard dataset configuration for our model

* Carsheets.io
  * Webscrape tool was used to access this dataset after permission was granted by
  the author for use. 
  * Unecessary 