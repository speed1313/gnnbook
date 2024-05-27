import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import click
import os

# グラフ畳み込みネットワークの定義
class GCN(torch.nn.Module):
    def __init__(self, in_d, mid_d, out_d, layer_num):
        super().__init__()
        self.conv1 = GCNConv(in_d, mid_d)
        self.inner_conv = GCNConv(mid_d, mid_d)
        self.conv2 = GCNConv(mid_d, out_d)
        self.layer_num = layer_num

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        for i in range(self.layer_num):
            x = self.inner_conv(x, edge_index)
            x = F.relu(x)

        emb = x.detach()
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), emb

class FFN(torch.nn.Module):
    def __init__(self, in_d, mid_d, out_d, layer_num):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_d, mid_d)
        self.inner_lin = torch.nn.Linear(mid_d, mid_d)
        self.lin2 = torch.nn.Linear(mid_d, out_d)
        self.layer_num = layer_num

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        for i in range(self.layer_num):
            x = F.relu(self.inner_lin(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), x

#@click.command()
#@click.option("--epoch", default=500)
#@click.option("--layer_num", default=0)
def main(epoch, layer_num, model_type="GCN"):
    if model_type == "GCN":
        dir = "figure/cora-classification/GCN"
    elif model_type == "FFN":
        dir = "figure/cora-classification/FFN"
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    n = dataset[0].num_nodes

    if model_type == "GCN":

        model = GCN(dataset.num_node_features, 16, dataset.num_classes, layer_num)
    elif model_type == "FFN":
        model = FFN(dataset.num_node_features, 16, dataset.num_classes, layer_num)

    data = dataset[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    def train(epoch):
        model.train()
        loss_list = []
        for epoch in range(epoch):
            optimizer.zero_grad()
            out = model(data)[0]
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        plt.close("all")
        plt.figure()
        plt.plot(loss_list)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()

    train(epoch)
    model.eval()
    pred = model(data)[0].argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    # TODO: histogram of prediction and true label
    plt.close("all")
    plt.figure()
    plt.hist(pred[data.test_mask].detach().numpy(), bins=range(8), alpha=0.5, label="pred")
    plt.hist(data.y[data.test_mask].detach().numpy(), bins=range(8), alpha=0.5, label="true")
    plt.legend()
    plt.show()
    

    print("Accuracy: {:.4f}".format(acc))

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(data.edge_index.shape[1]):
        G.add_edge(int(data.edge_index[0, i]), int(data.edge_index[1, i]))

    # ラベル色設定
    colors = [
        "#ff4b00",
        "#03af7a",
        "#005aff",
        "#4dc4ff",
        "#f6aa00",
        "#990099",
        "#804000",
    ]
    cs = [colors[y] for y in data.y]

    emb = model(data)[1].detach().numpy()
    tsne = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto")
    emb_tsne = tsne.fit_transform(emb)
    # distance between data[0] and others
    distance = np.linalg.norm(emb - emb[0], axis=1).mean()
    print("distance", distance)


    # GCN の埋め込み可視化
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(
        G,
        ax=ax,
        pos=emb_tsne,
        node_size=20,
        node_color=cs,
        labels={i: "" for i in range(n)},
        edge_color="#84919e",
        width=0.1,
    )

    plt.savefig(os.path.join(dir, f"emb_{layer_num}.png"))
    # plt.show()

    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(data.x)

    # 頂点特徴 X の可視化
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(
        G,
        ax=ax,
        pos=X_tsne,
        node_size=20,
        node_color=cs,
        labels={i: "" for i in range(n)},
        edge_color="#84919e",
        width=0.1,
    )
    # plt.savefig(f"figure/cora-classification/X.png")

    # plt.show()
    return acc, distance


if __name__ == "__main__":
    layer_num = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    model_type = "GCN"
    if model_type == "GCN":
        dir = "figure/cora-classification/GCN"
    elif model_type == "FFN":
        dir = "figure/cora-classification/FFN"
    if not os.path.exists(dir):
        os.makedirs(dir)

    accs = []
    dists = []
    for l in layer_num:
        acc, dist = main(epoch=1000, layer_num=l, model_type=model_type)
        accs.append(acc)
        dists.append(dist)
    print(accs)
    plt.figure()
    plt.plot(layer_num, accs)
    plt.xlabel("layer_num")
    plt.ylabel("accuracy")
    plt.xticks(layer_num)
    plt.tight_layout()
    plt.legend()

    plt.savefig(os.path.join(dir, "accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(layer_num, dists)
    plt.xlabel("layer_num")
    plt.ylabel("distance")
    plt.xticks(layer_num)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "distance.png"))
    # plt.show()
