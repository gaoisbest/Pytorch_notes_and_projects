import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sentiment_model import SentimentClassifier


parser = argparse.ArgumentParser(description='Train block hyper-parameter setting.')
parser.add_argument('--data', type=str, default='./data', help='Training data location.')
parser.add_argument('--epoch_num', type=int, default=50, help='Training epoch number.')
parser.add_argument('--log_interval', type=int, default=100, help='Training log interval')
parser.add_argument('--eval_interval', type=int, default=100, help='Evaluating log interval')
parser.add_argument('--cuda', help='CUDA is available.')
parser.add_argument('--save_path', type=str, default='./model.pt', help='Path to save the trained model.')
parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size.')
parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension.')
parser.add_argument('--num_units', type=int, default=256, help='The units number of hidden states.')
parser.add_argument('--atten_units', type=int, default=128, help='The units number of attention.')
parser.add_argument('--atten_hops', type=int, default=3, help='The heads number of attention.')
args = parser.parse_args()


def get_data():
    # Prepare the input data
    # From https://torchtext.readthedocs.io/en/latest/datasets.html

    from torchtext import data
    from torchtext import datasets

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    LABEL = data.LabelField(sequential=False)

    ds_train, ds_test = datasets.IMDB.splits(TEXT, LABEL)

    import random
    SEED = 1234
    ds_train, ds_valid = ds_train.split(random_state=random.seed(SEED))

    print('Train len: %d, valid len: %d, test len: %d' % (len(ds_train), len(ds_valid), len(ds_test)))

    TEXT.build_vocab(ds_train, max_size=25000)
    LABEL.build_vocab(ds_train)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (ds_train, ds_valid, ds_test), batch_size=args.batch_size, sort_key=lambda x: len(x.text), repeat=False
    )

    print('Train iter len: %d, valid iter len: %d, test iter len: %d' % (len(train_iterator), len(valid_iterator), len(test_iterator)))
    return train_iterator, valid_iterator, test_iterator, len(TEXT.vocab), len(ds_valid)


train_loader, valid_loader, test_loader, vocab_size, valid_len = get_data()
args.vocab_size = vocab_size
print('vocab size: %d' % args.vocab_size)

# model
model = SentimentClassifier(args)
# optimizer
optimizer = optim.Adam(model.parameters())
# loss
criterion = nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)


def validate():
    model.eval()
    total_correct = 0.0
    for idx, batch in enumerate(valid_loader):
        preds = model(batch.text.to(device))
        # torch.Size([64])
        prediction = torch.round(torch.sigmoid(preds.squeeze())).long()
        # torch.Size([64])
        res = batch.label.to(device).eq(prediction).sum().item()
        total_correct = total_correct + res
    # print('total_correct: %d, valid_len: %d, ratio: %.2f' % (total_correct, valid_len, total_correct/valid_len))
    return total_correct / valid_len


def train():
    model.train()
    tmp_loss = 0.0
    best_accuracy = 0.0
    for epoch in range(args.epoch_num):
        for idx, batch in enumerate(train_loader):
            # zero gradient for each batch
            optimizer.zero_grad()
            preds = model(batch.text.to(device))

            loss = criterion(preds.squeeze(), batch.label.to(device).float())
            tmp_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (idx+1) % args.log_interval == 0:
                print('Epoch: [%d:%d], batch: [%d:%d], loss: %.3f, best accuracy: %.2f' %
                      (epoch, args.epoch_num, idx, len(train_loader), tmp_loss/args.log_interval, best_accuracy))
                tmp_loss = 0.0

            if (idx+1) % args.eval_interval == 0:
                tmp_accuracy = validate()
                if tmp_accuracy >= best_accuracy:
                    best_accuracy = tmp_accuracy
                    torch.save(model.state_dict(), args.save_path)
                    print('New model saved with accuracy: %.2f' % best_accuracy)
                else:
                    print('Old model with accuracy: %.2f' % best_accuracy)
                model.train()


if __name__ == '__main__':
    train()
