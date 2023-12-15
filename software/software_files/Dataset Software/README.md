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
  * Unecessary categories were remove and only the following data information was retained
    * Make, Model, Year, Curb Weight, Car Type
  * To avoid misclassifying weights and creating a sparse column, we took the following
  steps:
    * Query both the CompCars and Carsheets.io dataset to find matching Make and Model pairs
      * If their is an identified pair, directly transfer the weight from carsheets to CompCars
      * Else create a seperate dataframe containing all uncompatible models
    * Query all defined individual curb weights and group them by CompCars Car Type
    * Remove duplicate entries resulting from identical models by weight over years
    * Find the average weight for the remaning classified Car Types
    * Used the averge weight for each Car Type to fill missing information in our
    avg_curb_weight column.

### Final Dataset Config:
The following displays the final configuration of our dataset contained in the final_data_config.csv file
in our data folder:

|car_type_id|x_1               |y_1               |x_2               |y_2               |year|make_name       |model_name               |car_type_name      |make_id|model_id|avg_curb_weight|
|-----------|------------------|------------------|------------------|------------------|----|----------------|-------------------------|-------------------|-------|--------|---------------|
|0          |0.5022026431718062|0.5678913738019169|0.7951541850220264|0.8642172523961661|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.5099009900990099|0.5669856459330144|0.5291529152915292|0.6140350877192983|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.5198237885462555|0.5638977635782748|0.8303964757709251|0.8083067092651757|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.4906387665198238|0.5990415335463258|0.8623348017621145|0.7603833865814696|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |
|0          |0.4900881057268722|0.5862619808306709|0.7687224669603524|0.5239616613418531|2009|Honda           |Odyssey                  |MPV                |100    |209     |4441.0         |

