import csv

with open('Data/fracture_k_sequence_inter_all.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('Data/fracture_k_sequence_inter_all2.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)