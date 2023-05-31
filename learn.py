from utils import *
import torch
import copy

def train(encoder, dist_encoder, prototype, data, optimizer, criterion, args):
    encoder.train()

    support, query = episodic_generator(
        data, args.episodic_samp, args.classes, data.x.size(0))

    embedding = encoder(data)

    support_embed = [embedding[support[i]] for i in range(len(args.classes))]

    query_embed = [embedding[query[i]] for i in range(len(args.classes))]
    query_size = [query_embed[i].size() for i in range(len(query_embed))]

    query_embed = torch.stack(query_embed, dim=0)

    proto_embed = [prototype(support_embed[i])
                   for i in range(len(args.classes))]

    proto_embed = torch.stack(proto_embed, dim=0)  # C*D

    query_dist_embed = dist_encoder(query_embed, proto_embed, args.classes)
    proto_dist_embed = dist_encoder(proto_embed, proto_embed, args.classes)

    logits = torch.log_softmax(
        torch.mm(query_dist_embed, proto_dist_embed), dim=1)

    loss1 = criterion(logits, args.classes)

    # topo
    if(args.ssl == 'yes'):
        dist_embed = dist_encoder(embedding, proto_embed, args.classes)
        loss3 = torch.mean((dist_embed[data.edge_index[0]] * args.deg_inv_sqrt[data.edge_index[0]].view(-1, 1) -
                            dist_embed[data.edge_index[1]] * args.deg_inv_sqrt[data.edge_index[1]].view(-1, 1))**2)

        class_sim = cos_sim_pair(proto_embed)
        loss2 = (torch.sum(class_sim) - torch.trace(class_sim)) / \
            ((class_sim.size(0)**2 - class_sim.size(0)) / 2)
    else:
        loss3 = 0
        loss2 = 0

    loss = loss1 + args.lamb1 * loss2 + args.lamb2 * loss3

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test(encoder, dist_encoder, prototype, data, args):
    encoder.eval()

    with torch.no_grad():
        embedding = encoder(data)

        support, query = episodic_generator(
            data, 1, args.classes, data.x.size(0))  # take all samples in that class
        support_embed = [embedding[support[i]]
                         for i in range(len(args.classes))]

        proto_embed = [prototype(support_embed[i])
                       for i in range(len(args.classes))]

        proto_embed = torch.stack(proto_embed, dim=0)  # C*D

        f1, f1w, acc = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            y = data.y[mask]

            query_embed = embedding[mask]  # N*D
            # query_dist = torch.cdist(query_embed, proto_embed, p = 2) #N*D, C*D --> N*C
            query_dist_embed = dist_encoder(
                query_embed, proto_embed, args.classes)
            proto_dist_embed = dist_encoder(
                proto_embed, proto_embed, args.classes)
            # logits = torch.softmax(-query_dist, dim = 1) #N*C
            logits = torch.log_softmax(
                torch.mm(query_dist_embed, proto_dist_embed), dim=1)

            pred = logits.max(dim=1)[1]

            acc.append(pred.eq(y).sum().item() / mask.sum().item())
            f1.append(f1_score(y.tolist(), pred.tolist(), labels=np.arange(
                0, len(args.classes)), average=None, zero_division=0))
            f1w.append(f1_score(y.tolist(), pred.tolist(), labels=np.arange(
                0, len(args.classes)), average='weighted', zero_division=0))

    return f1, f1w, acc
