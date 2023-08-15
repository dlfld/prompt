import csv
import os
import re


def get_loss_file():
    folder_path = 'ud_ch_bert_medbert_bart/'

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            pattern = r'bert_large_chinese_[0-9]_1.csv'
            if re.match(pattern, file):
                file_path = os.path.join(root, file)
                with open(file_path,"r") as f:
                    content = f.readlines()

            with open("resres.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerows(content)
