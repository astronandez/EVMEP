import os
import csv
import pandas as pd

text_path = 'YoloFormatData/yolo_formated_data.txt'
out_path = 'YoloFormatData/yolo_car_type_data.csv'

lines = []
with open(text_path, 'r') as input_file:
    next(input_file)
    # Read lines from the text file
    for line in input_file:
        column = line.strip().split(',')
        column[1] = int(column[1])
        if(column[1] != 0):
            # lines.append([column[0], (column[1]-1)])
            lines.append([column[0], column[1]])


# Open the CSV file for writing
with open(out_path, 'w', newline='') as output_file:
    # Create a CSV writer
    csv_writer = csv.writer(output_file)

    # Write each line to the CSV file
    for line in lines:
        # Split the line into fields based on spaces or tabs
        # field = line.strip().split(' ')  # Change '\t' to ' ' if the values are separated by spaces
        # Write the fields to the CSV file
        csv_writer.writerow(line)

print(f"Conversion complete. CSV file saved at {out_path}")