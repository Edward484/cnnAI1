import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from DeepHallucinationDataset import create_dataloader
from model import ModelCNN
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main(data_dir, image_txt, image_dir, eval_txt, test_txt, test_dir):
    learning_rate = 0.01

    # number of epochs
    no_epochs = 30

    # training model on local cpu
    device = torch.device("cpu")

    # create dataloader
    train_dataloader = create_dataloader(data_dir, image_txt, image_dir, role="Train", shuffle=True, batch_size=16)
    eval_dataloader = create_dataloader(data_dir, eval_txt, image_dir, role="Eval", shuffle=True, batch_size=16)

    # test_dataloader contains on the second positions the filenames and on first the files
    test_dataloader = create_dataloader('../data', '../data/test.txt', '../data/test', role="Test", shuffle=False,
                                        batch_size=16)

    # initialize the model
    cnn_model = ModelCNN()
    cnn_model = cnn_model.to(device)

    # initialize the loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # initialize SGD optimzer with momentum 0.9 to help accelerate converging
    optimizer = SGD(params=cnn_model.parameters(), lr=learning_rate, momentum=0.9)

    # initialized the reducer so that when loss start hitting a plateau, it brings the LR down so that it will better converge
    # verbose is for printing in console the update
    # patience = 4 because the number of epochs it's pretty low
    # factor = 0.05 (default is 0.1) for smaller but more frequent adjustments
    # this also helps to prevent over fit
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, factor=0.1)

    for epoch in range(no_epochs):
        loss_train = train_model(cnn_model, device, loss_function, train_dataloader, optimizer, epoch)
        acc_eval, loss_eval = evaluate_model(cnn_model, device, loss_function, eval_dataloader, epoch)
        scheduler.step(loss_eval)
        print_stats(loss_train, acc_eval, loss_eval, epoch)

    # after the training in complete, test the model on the test set
    test_model(cnn_model, device, test_dataloader)

    return cnn_model


def train_model(cnn_model, device, loss_function, dataloader, optimizer, current_epoch):
    cnn_model.train()

    # keeps the sum of all losses
    all_loss = 0.

    for images, labels in tqdm(dataloader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        # vector of raw, unprocessed predicted labels
        logits = cnn_model(images)

        # Apply the loss function on predicted result and on real result to get the loss
        loss = loss_function(logits, labels)

        # cleans gradients
        optimizer.zero_grad()

        # make the backward step. Update gradients with the new loss
        loss.backward()

        # Update the weights and biases
        optimizer.step()

        all_loss += loss.item() * len(images)

    # mean value of the loss of this train set
    mean_loss = all_loss / len(dataloader.dataset)
    return mean_loss


def evaluate_model(cnn_model, device, loss_function, dataloader, current_epoch):
    cnn_model.eval()

    all_loss = 0.

    #keeps track of the number of true predictions
    no_true_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Eval"):
            images = torch.tensor(images).to(device)
            labels = torch.tensor(labels).to(device)

            # vector of raw, unprocessed predicted labels
            logits = cnn_model(images)

            # Apply the loss function on predicted result and on real result to get the loss
            loss = loss_function(logits, labels)

            scores = softmax(logits, dim=1)

            prediction_label = torch.argmax(scores, dim=1)

            no_true_predictions += (prediction_label == labels).sum()

            all_loss += loss.item() * len(images)


    #calculates the accuracy for eatch epoch
    accuracy = no_true_predictions / len(dataloader.dataset)

    all_loss = all_loss / len(dataloader.dataset)
    return accuracy, all_loss


def test_model(cnn_model, device, dataloader):
    cnn_model.eval()

    file = open("test_result_cnn2.txt", "a")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Test"):
            images = torch.tensor(images).to(device)

            # vector of raw, unprocessed predicted labels
            logits = cnn_model(images)

            scores = softmax(logits, dim=1)

            # get the best prediction from the different classes
            prediction_label = torch.argmax(scores, dim=1)

            # write to file
            # pair[0] is the prediction and pair[1] is the filename that was kept in labels
            for pair in zip(prediction_label.numpy(), labels):
                file.write(f"{pair[1]},{pair[0]}\n")

    file.close()


# printing statistics in console after every epoch to keep track of under-fitting or over-fitting
def print_stats(loss_train, acc_eval, loss_eval, epoch_number):
    print(
        f"\n-------------------------\nAccuracy: {acc_eval}, loss train: {loss_train}, loss eval: {loss_eval}, epoch: {epoch_number}\n-------------------------\n")


def predict():
    pass


if __name__ == '__main__':
    DATA_DIR = '../data'
    IMAGE_TXT = '../data/train.txt'
    IMAGE_DIR = '../data/train+validation'
    EVAL_TXT = '../data/validation.txt'
    TEST_TXT = '../data/test.txt'
    TEST_DIR = '../data/test'

    model = main(DATA_DIR, IMAGE_TXT, IMAGE_DIR, EVAL_TXT, TEST_TXT, TEST_DIR)
