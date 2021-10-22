from dataset import *
from utils import *
from model import *
from learn import *

import argparse
import random
from torch import tensor
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--shuffle', type=str, required=True)
parser.add_argument('--encoder', type=str, required=True)
parser.add_argument('--episodic_samp', type=float, required=True)
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('--imb_ratio', type=float, default=10)
parser.add_argument('--label_prop', type=str, required=True)
parser.add_argument('--eta', type=float, required=True)
parser.add_argument('--ssl', type=str, required=True)

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lamb1', type=float, default=1)
parser.add_argument('--lamb2', type=float, default=1)

parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--early_stopping', type=int, default=0)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--weight_decay', type=float, default=5e-4)


args = parser.parse_args()

# load data
data, class_sample_num, args.num_classes, args.num_features, args.c_train_num, args.classes = load_data(
    args.dataset, args.imb_ratio)
#print the number of training data for each class
print('The number of training data for each class', args.c_train_num)


if(args.ssl == 'yes'): #if using self-supervised learning component
    deg_inv_sqrt = deg(data.edge_index, data.x)


F1 = np.zeros((args.runs, args.num_classes), dtype=float)
F1_weight = np.zeros(args.runs, dtype=float)
F1_micro = np.zeros(args.runs, dtype=float)
pbar = tqdm(range(args.runs), unit='run')

for count in pbar:
    random.seed(args.seed + count)
    np.random.seed(args.seed + count)
    torch.manual_seed(args.seed + count)
    torch.cuda.manual_seed(args.seed + count)

    data = shuffle(data, args.num_classes, args.c_train_num)

    if(args.label_prop == 'yes'):
        y_prop = label_prop(data.edge_index, data.train_mask,
                            args.c_train_num, data.y, epochs=20)

        # plot for tuning threshold \eta in imbalanced label propagation
        # y_aug_trail(data.y, y_prop, args.c_train_num, data.val_mask)

        y_aug, train_mask = sample(
            data.train_mask, args.c_train_num, y_prop, data.y, eta=args.eta)

        data.y_aug, data.train_mask = y_aug, train_mask

    if(args.label_prop == 'no'):
        data.y_aug, data.train_mask = data.y, data.train_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, proto = data.to(device), prototype().to(device)

    if(args.ssl == 'yes'):
        args.deg_inv_sqrt = deg_inv_sqrt.to(device)

    if(args.encoder == 'GCN'):
        encoder = GCN(args).to(device)

    dist_encoder = dist_embed(args).to(device)
    args.classes = args.classes.to(device)


    model_param_group = []
    if(args.encoder == 'GCN'):
        model_param_group.append(
            {'params': encoder.conv1.parameters(), 'lr': 1e-2, 'weight_decay': 5e-4})
        model_param_group.append(
            {'params': encoder.conv2.parameters(), 'lr': 1e-2, 'weight_decay': 0})

    model_param_group.append(
        {'params': dist_encoder.lin.parameters(), 'lr': 1e-2, 'weight_decay': 0})

    optimizer = torch.optim.Adam(model_param_group)

    criterion = torch.nn.NLLLoss()

    best_val_f1_mean = 0
    val_f1_history = []

    for epoch in range(args.epochs):
        train(encoder, dist_encoder, proto, data, optimizer, criterion, args)
        f1, f1w, accs = test(encoder, dist_encoder, proto, data, args)

        train_f1, val_f1, tmp_test_f1 = f1[0], f1[1], f1[2]
        train_f1w, val_f1w, tmp_test_f1w = f1w[0], f1w[1], f1w[2]
        train_acc, val_acc, tmp_test_acc = accs[0], accs[1], accs[2]

        if np.mean(val_f1) > best_val_f1_mean:
            best_val_f1_mean = np.mean(val_f1)
            test_f1 = tmp_test_f1
            test_acc = tmp_test_acc
            test_f1w = tmp_test_f1w

            # torch.save(encoder.state_dict(), 'encoder.pkl')
            # torch.save(dist_encoder.state_dict(), 'dist_encoder.pkl')

        val_f1_history.append(np.mean(val_f1))
        if args.early_stopping > 0 and epoch > args.epochs // 2:
            tmp = tensor(val_f1_history[-(args.early_stopping + 1): -1])
            if np.mean(val_f1) < tmp.mean().item():
                break


    F1[count] = test_f1
    F1_micro[count] = test_acc
    F1_weight[count] = test_f1w

    print('F1-macro:', np.sum(F1) / ((count + 1) * args.num_classes), 'F1-weight:', np.sum(F1_weight) / (count+1), 'F1-micro:', np.sum(F1_micro) / (count + 1))

print('Acc for each class:', np.mean(F1, axis = 0))
print('F1-macro: ', np.mean(F1))
print('F1-weight: ', np.mean(F1_weight))
print('F1-micro: ', np.mean(F1_micro))
