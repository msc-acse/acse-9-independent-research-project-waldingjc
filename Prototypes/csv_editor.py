import csv
with open('Data/fracture_k_sequence_set_all.csv', 'r') as in_file:
    reader = csv.reader(in_file)
    with open('Data/fracture_k_sequence_set_all_edit.csv', 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        for row in reader: # 4 5 6 8 9 10
            error = False
            for i, ele in enumerate(row):
                ele_num = float(ele)
                if i > 9:
                    if ele_num == 0:
                        error = True
                if abs(ele_num) > 1e8:
                    error = True
            if row[-4] <= "0":
                error = True
            if row[-8] <= "0":
                error = True
            #if (row[4]>="0") and (row[5]!="0") and (row[6]!="0") and (row[8]>="0") and (row[9]!="0") and (row[10]!="0"):
                #writer.writerow(row)
            if not error:
                writer.writerow(row)