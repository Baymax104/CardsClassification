import os
import torch
from tqdm import tqdm
from dataset import PokerDataset
from torch import optim, nn
from net import ClassifyNet
import parameters as params
from torchmetrics import classification as metrics
import torchmetrics
from recorder import Recorder


def TrainNet():
    print('=' * 40 + f'Training in {params.DEVICE}' + '=' * 40)

    # datasets
    train_loader = PokerDataset(dataset='train').to_loader()
    valid_loader = PokerDataset(dataset='valid').to_loader()

    # metrics
    train_acc_metric = metrics.MulticlassAccuracy(num_classes=params.CLASSES).to(params.DEVICE)
    valid_acc_metric = metrics.MulticlassAccuracy(num_classes=params.CLASSES).to(params.DEVICE)
    train_f1_metric = metrics.MulticlassF1Score(num_classes=params.CLASSES).to(params.DEVICE)
    valid_f1_metric = metrics.MulticlassF1Score(num_classes=params.CLASSES).to(params.DEVICE)

    # model component
    model = ClassifyNet().to(params.DEVICE)
    criterion = nn.CrossEntropyLoss().to(params.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=params.LEARN_RATE, weight_decay=params.WEIGHT_DECAY)

    # record
    recorder = Recorder('train')


    # ============================ Train Start ============================
    print('=' * 40 + 'Train Start' + '=' * 40)

    # Epoch
    for epoch in range(params.EPOCH):

        # loss list records every loss value to calculate its average value in every epoch
        train_loss_list = []
        valid_loss_list = []

        # ============================ Train ============================
        model.train()
        looper = tqdm(train_loader, total=len(train_loader), desc=f'[Epoch: {epoch + 1}] Training', colour='green')
        for batch in looper:
            data, target = batch
            data, target = data.to(params.DEVICE, non_blocking=True), target.to(params.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            train_loss_list.append(loss.item())

            train_acc_metric(output, target)
            train_f1_metric(output, target)

            loss.backward()
            optimizer.step()

        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_acc = train_acc_metric.compute().item()
        train_f1 = train_f1_metric.compute().item()

        train_acc_metric.reset()
        train_f1_metric.reset()



        # ============================ Validate ============================
        model.eval()
        with torch.no_grad():
            looper = tqdm(valid_loader, total=len(valid_loader), desc=f'[Epoch: {epoch + 1}] Validating', colour='green')
            for batch in looper:
                data, target = batch
                data, target = data.to(params.DEVICE), target.to(params.DEVICE)

                output = model(data)
                loss = criterion(output, target)
                valid_loss_list.append(loss.item())

                valid_acc_metric(output, target)
                valid_f1_metric(output, target)

            # convert to proportion by dividing
            valid_loss = sum(valid_loss_list) / len(valid_loss_list)
            valid_acc = valid_acc_metric.compute().item()
            valid_f1 = valid_f1_metric.compute().item()

            valid_acc_metric.reset()
            valid_f1_metric.reset()

        # ============================ Record ============================
        recorder.add_train_record({
            'epoch': epoch,
            'train': [train_loss, train_acc, train_f1],
            'valid': [valid_loss, valid_acc, valid_f1]
        })

    recorder.close()
    # end epoch

    # ============================ Save ============================
    recorder.save(model)
    print('Save Successfully!')

    print('=' * 40 + 'Train Done' + '=' * 40)
    # ============================ Train Done ============================


def TestNet():
    print('=' * 40 + f'Testing in {params.DEVICE}' + '=' * 40)

    # dataset
    test_loader = PokerDataset(dataset='test').to_loader()

    # load model
    model_id = '05-30_02-00'
    state = torch.load(os.path.join(params.SAVE_DIR, model_id, f'model-{model_id}.pt'), map_location=torch.device('cpu'))
    model = ClassifyNet()
    model.load_state_dict(state)
    model = model.to(params.DEVICE)

    # metrics
    metric = torchmetrics.MetricCollection({
        'acc': metrics.MulticlassAccuracy(num_classes=params.CLASSES),
        'f1': metrics.MulticlassF1Score(num_classes=params.CLASSES),
        'precision': metrics.MulticlassPrecision(num_classes=params.CLASSES),
        'recall': metrics.MulticlassRecall(num_classes=params.CLASSES),
        'confusion': metrics.MulticlassConfusionMatrix(num_classes=params.CLASSES)
    }).to(params.DEVICE)

    # record
    recorder = Recorder('test', record_id=model_id)



    # ============================ Test Start ============================
    print('=' * 40 + 'Test Start' + '=' * 40)

    model.eval()
    with torch.no_grad():
        looper = tqdm(test_loader, total=len(test_loader), desc=f'Model Testing', colour='green')
        for batch in looper:
            data, target = batch
            data, target = data.to(params.DEVICE), target.to(params.DEVICE)

            output = model(data)
            metric(output, target)


    metrics_dict = metric.compute()
    metric.reset()

    # convert tensor to ndarray
    for index, value in metrics_dict.items():
        metrics_dict[index] = value.cpu().numpy()

    # ============================ Record ============================
    recorder.add_test_record(metrics_dict)
    recorder.save()

    print('=' * 40 + 'Test Done' + '=' * 40)
    # ============================ Test Done ============================













if __name__ == '__main__':
    torch.manual_seed(params.RANDOM_SEED)
    # TrainNet()
    TestNet()
