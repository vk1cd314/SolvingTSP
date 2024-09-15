import dgl
dataset = dgl.data.CSVDataset('./generated-data')
g = dataset

print(dataset[0])

