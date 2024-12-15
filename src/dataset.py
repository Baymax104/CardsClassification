from PIL import Image
from os import path
import pandas as pd
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
import parameters as params
from dataloader import Loader


class PokerDataset(data.Dataset):

    def __init__(self, dataset=None):
        super(PokerDataset, self).__init__()
        self.data = None  # data of sample
        self.target = None  # target of sample

        self.root_dir: str = params.ROOT_DIR
        self.data_dirname = dataset
        self.target_name: str = params.TARGET_FILENAME

        self.data_table = pd.read_csv(path.join(self.root_dir, self.target_name))
        self.data_table = self.data_table[self.data_table['dataset'] == dataset]

        self.load()


    def load(self):

        transformer = transforms.ToTensor()

        # load data
        data = []
        target = []
        for _, sample in tqdm(self.data_table.iterrows(), desc=f'loading {self.data_dirname}', colour='green'):
            # load data
            image = path.join(self.root_dir, sample['filepaths'])
            image = Image.open(image)
            image = transformer(image)
            data.append(image)

            # load target
            target.append(sample['target'])

        self.data = data
        self.target = target

        print(f'loaded data size: {len(data)}, loaded target size: {len(target)}')

        # check sample number
        if len(data) != len(target):
            raise Exception('sample number error!')


    def __getitem__(self, index):
        return self.data[index], self.target[index]


    def __len__(self):
        return len(self.target)


    def to_loader(self) -> Loader:
        return Loader(self)

