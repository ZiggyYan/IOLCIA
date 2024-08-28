import csv
import os
def read_csv(filename):
    data = []
    with open(filename,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

# 检查文件是否存在
def check_file_exists(file_path):
    return os.path.exists(file_path)
if __name__ == '__main__':
    train_csv_name='./Dataset/train_label.csv'
    valid_csv_name='./Dataset/val_label.csv'
    train_dir='./Dataset/image/train/'
    valid_dir='./Dataset/image/valid/'
    train_data=read_csv(train_csv_name)
    valid_data=read_csv(valid_csv_name)

    for row in train_data:
        flag = check_file_exists(os.path.join(train_dir,row[1],row[0]))
        if not flag:
            print(os.path.join(train_dir,row[1],row[0])+" NOT FOUND")
    for row in valid_data:
        flag = check_file_exists(os.path.join(valid_dir,row[1],row[0]))
        if not flag:
            print(os.path.join(valid_dir,row[1],row[0])+" NOT FOUND")

