import os
import time
import parameters as params
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
import torch
import pandas as pd
import pickle


class Recorder:

    # for train
    total_acc = []
    total_loss = []
    total_f1 = []

    # for test
    scalar_dict = {}
    curve_dict = {}

    def __init__(self, record_type, record_id=None):
        self.type = record_type
        self.id = record_id

        if record_type == 'train':
            self.writer = SummaryWriter(log_dir=params.LOG_DIR)
            self.table = PrettyTable()
            self.table.field_names = ['epoch', 'train loss', 'valid loss', 'train accuracy', 'valid accuracy']
        elif record_type == 'test' and record_id is not None:
            self.test_table = PrettyTable()
            self.test_table.field_names = ['Accuracy', 'Macro-F1 Score', 'Precision', 'Recall']
        else:
            raise Exception('Record Type Error!')


    def add_train_record(self, record_dict):
        if self.type != 'train':
            raise Exception('This recorder is not for train, check parameter record_type')

        epoch = record_dict['epoch']
        train_loss, train_acc, train_f1 = record_dict['train']
        valid_loss, valid_acc, valid_f1 = record_dict['valid']
        self.table.add_row([epoch + 1, train_loss, valid_loss, train_acc, valid_acc])

        if (epoch + 1) % 20 == 0:
            print(self.table)
            self.table.clear_rows()

        self.writer.add_scalars('Loss', {'train': train_loss, 'valid': valid_loss}, global_step=epoch + 1)
        self.writer.add_scalars('Accuracy', {'train': train_acc, 'valid': valid_acc}, global_step=epoch + 1)
        self.writer.add_scalars('F1 Score', {'train': train_f1, 'valid': valid_f1}, global_step=epoch + 1)

        self.total_loss.append({'epoch': epoch + 1, 'train': train_loss, 'valid': valid_loss})
        self.total_acc.append({'epoch': epoch + 1, 'train': train_acc, 'valid': valid_acc})
        self.total_f1.append({'epoch': epoch + 1, 'train': train_f1, 'valid': valid_f1})


    def add_test_record(self, metrics_dict):
        if self.type != 'test':
            raise Exception('This recorder is not for test, check parameter record_type')

        acc, f1 = metrics_dict['acc'], metrics_dict['f1']
        precision, recall = metrics_dict['precision'], metrics_dict['recall']
        self.scalar_dict = {'acc': acc, 'f1': f1, 'precision': precision, 'recall': recall}
        self.curve_dict = {'confusion': metrics_dict['confusion']}

        self.test_table.add_row([acc, f1, precision, recall])
        print(self.test_table)



    def save(self, model=None):
        if self.type == 'train' and model is None:
            raise Exception('This recorder is for train, it must save a model')

        if self.type == 'train':
            record_id = time.strftime('%m-%d_%H-%M', time.localtime())
            dir_path = os.path.join(params.SAVE_DIR, record_id)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            model_filename = f'model-{record_id}.pt'

            param = {
                'batch': params.BATCH_SIZE,
                'epoch': params.EPOCH,
                'lr': params.LEARN_RATE,
                'decay': params.WEIGHT_DECAY,
                'seed': params.RANDOM_SEED
            }

            loss = pd.DataFrame(self.total_loss)
            acc = pd.DataFrame(self.total_acc)
            f1 = pd.DataFrame(self.total_f1)
            param = pd.DataFrame(param, index=[0])

            # save
            torch.save(model.state_dict(), os.path.join(dir_path, model_filename))
            loss.to_csv(os.path.join(dir_path, 'loss.csv'), index=False)
            acc.to_csv(os.path.join(dir_path, 'acc.csv'), index=False)
            f1.to_csv(os.path.join(dir_path, 'f1-score.csv'), index=False)
            param.to_csv(os.path.join(dir_path, 'params.csv'), index=False)

        elif self.type == 'test':
            dir_path = os.path.join(params.SAVE_DIR, self.id, 'test')
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            # save scalar
            scalar_metrics = pd.DataFrame(self.scalar_dict, index=[0])
            scalar_metrics.to_csv(os.path.join(dir_path, 'scalar.csv'), index=False)

            # save curve
            with open(os.path.join(dir_path, 'curve.pkl'), 'wb') as file:
                pickle.dump(self.curve_dict, file)

        print("Save Successfully!")


    def close(self):
        if self.type == 'train':
            self.writer.close()
