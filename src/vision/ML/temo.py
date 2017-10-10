import csv
with open('plant_db.csv', newline='') as csvfile:
    dataset = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in dataset:
        print(', '.join(row))