import os
import csv

text_path = r'Z:\CompCars\data\misc\attributes.txt'
out_path = 'YoloFormatData/test.csv'

with open(text_path, 'r') as input_file:
    # Read lines from the text file
    lines = input_file.readlines()

# Open the CSV file for writing
with open(out_path, 'w', newline='') as output_file:
    # Create a CSV writer
    csv_writer = csv.writer(output_file)

    # Write each line to the CSV file
    for line in lines:
        # Split the line into fields based on spaces or tabs
        fields = line.strip().split(' ')  # Change '\t' to ' ' if the values are separated by spaces

        # Write the fields to the CSV file
        csv_writer.writerow(fields)

print(f"Conversion complete. CSV file saved at {out_path}")