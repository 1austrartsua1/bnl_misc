
import torch
from torch.utils.data import Dataset, DataLoader

class MyRandomDataSet(Dataset):
    def __init__(self,dim1,dim2,numDataPoints,numLabels,numColorChannels=3,seedToUse = None):
        self.dim1 = dim1
        self.dim2 = dim2
        self.numDataPoints = numDataPoints
        self.seed = seedToUse
        self.numLabels = numLabels
        if self.seed:
            torch.seed(self.seed)
        self.images = torch.randn(numDataPoints,numColorChannels,dim1,dim2)
        self.labels = torch.randint(0,numLabels,(numDataPoints,))

    def __len__(self):
        return self.numDataPoints


    def __getitem__(self,idx):
        return (self.images[idx],self.labels[idx])


if __name__ == "__main__":
    dim1 = 10
    dim2 = 20
    n = 100
    nlabels = 10
    aRandomDataset = MyRandomDataSet(dim1,dim2,n,nlabels)
    myLoader = DataLoader(aRandomDataset,batch_size=10,shuffle=True)
    for batchindex,(data,target) in enumerate(myLoader):
        print(batchindex)
        print(data.shape)
        print(target.shape)
    print('done')
