import pandas as pd
import numpy as np


classes = ['Convertible', 'Crossover', 'Fastback', 'Hardtop Convertible', 'Hatchback', 'MPV', 'Minibus', 'Pickup Truck', 'SUV', 'Sedan', 'Sports', 'Wagon']

data = pd.read_csv('/Users/Marc Hernandez/Documents/UCLA/ECE 202A/EVMEP/data/csv_dataset_files/avg_weight_on_per_class.csv')
def estimate_vechicle_weight(car_type_label):
    estimated_mass = 0
    if car_type_label == classes[0]:
        estimated_mass = data['avg_class_weight'][0]
    elif car_type_label == classes[1]:
        estimated_mass = data['avg_class_weight'][1]
    elif car_type_label == classes[2]:
        estimated_mass = data['avg_class_weight'][2]
    elif car_type_label == classes[3]:
        estimated_mass = data['avg_class_weight'][3]
    elif car_type_label == classes[4]:
        estimated_mass = data['avg_class_weight'][4]
    elif car_type_label == classes[5]:
        estimated_mass = data['avg_class_weight'][5]
    elif car_type_label == classes[6]:
        estimated_mass = data['avg_class_weight'][6]
    elif car_type_label == classes[7]:
        estimated_mass = data['avg_class_weight'][7]
    elif car_type_label == classes[8]:
        estimated_mass = data['avg_class_weight'][8]
    elif car_type_label == classes[9]:
        estimated_mass = data['avg_class_weight'][9]
    elif car_type_label == classes[10]:
        estimated_mass = data['avg_class_weight'][10]
    elif car_type_label == classes[11]:
        estimated_mass = data['avg_class_weight'][11]
    else:
        estimated_mass = 0

    return estimated_mass

