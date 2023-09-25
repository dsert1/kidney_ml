import torch
import csv
import numpy as np
import pandas as pd
from data import dataset

class KidneyDataset(torch.utils.data.Dataset):

    def __init__(self, csv_dir, label, features=None):

        content = self.read_csv(csv_dir)
        print("csv content: ", content)
        # if features is None:
        #     features = list(content[0].keys())  # All columns except the label
        #     features.remove(label)
        print("dataset.py features: ", features)
        self.features = features
        # self.features = features
        self.label = label
        # print("self.features: ", self.features)
        print("self.label: ", self.label)
        self.content = self.filter_incomplete_cases(content)
        print("self.content after filtering: ", self.content[0][self.label])
        self.x = np.array([[float(row[k]) for k in self.features] for row in self.content], dtype=np.float32)
        self.y = np.array([1 if int(row[self.label]) == 1 else 0 for row in self.content], dtype=np.float32) # 1 is 1 in the raw dataset, 2 is 0
        print("self.x: ", self.x)
        print("self.y: ", self.y)
    def read_csv(self, csv_file):

        content = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content.append(row)
        return content

    def filter_incomplete_cases(self, content):
        # print("content[0], ", content[0])
        # print("")
        filtered_content = []
        for row in content:
            complete = True
            for key in self.features:
                # print("key: ", key)
                if row[key] == '':
                    complete = False
            # print("row: ", row)
            # print("complete?: ", complete)
            # print("row[self.label]: ", row[self.label])
            if complete and row[self.label] != '':
                filtered_content.append(row)
        # print("filtered content: ", filtered_content)
        return filtered_content

    def __len__(self):

        return len(self.content)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]

    def input_length(self):

        return len(self.__getitem__(0)[0])

    @property
    def all(self):

        return self.x, self.y

if __name__ == "__main__":

    df = pd.read_csv(dataset)
    data = KidneyDataset(
        csv_dir='./data/kidney_data.csv',
        label='DiseaseOutcome'  # replace with the actual column name representing the outcome in your dataset
    )
