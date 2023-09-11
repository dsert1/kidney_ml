import torch
from dataset import KidneyDataset

def get_dataloader(csv_dir, label, batch_size=32, shuffle=True, features=None):
    print("Features:", features)
    print("Label:", label)

    dataset = KidneyDataset(csv_dir, label, features)
    print("line 8 dataset: ", dataset)
    print("batch size: ", batch_size)
    print("shuffle: ", shuffle)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

if __name__ == "__main__":
    LABEL_COLUMN = 'GDvsTI'
    dataloader = get_dataloader('./data/kidney_data.csv', LABEL_COLUMN, batch_size=4)
    for batch_input, batch_label in dataloader:
        print(batch_input)
        print(batch_label)
