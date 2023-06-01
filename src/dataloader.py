from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import parameters as params


class Loader(DataLoader):

    def __init__(self, dataset):
        super(Loader, self).__init__(
            dataset,
            batch_size=params.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
