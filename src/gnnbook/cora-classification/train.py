import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    n = dataset[0].num_nodes
    # dataset[0] attribute
    print(dataset[0])
    # dataset[0] method list
    print(dir(dataset[0]))
    print(dataset[0].x.shape)

    model = GCN(dataset.num_node_features, 16, dataset.num_classes, 20)

    data = dataset[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)

    def train(epoch):
        model.train()
        for epoch in range(epoch):
            optimizer.zero_grad()
            out = model(data)[0]
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    train(500)
    model.eval()
    pred = model(data)[0].argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print("Accuracy: {:.4f}".format(acc))

    print(data.edge_index.shape)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(data.edge_index.shape[1]):
        G.add_edge(int(data.edge_index[0, i]), int(data.edge_index[1, i]))

    # ラベル色設定
    colors = [
        '#ff4b00',
        '#03af7a',
        '#005aff',
        '#4dc4ff',
        '#f6aa00',
        '#990099',
        '#804000'
    ]
    cs = [
        colors[y] for y in data.y
    ]

    emb = model(data)[1].numpy()
    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate="auto")
    emb_tsne = tsne.fit_transform(emb)


    # GCN の埋め込み可視化
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(G, ax=ax, pos=emb_tsne, node_size=20, node_color=cs, labels={i: '' for i in range(n)}, edge_color='#84919e', width=0.1)

    plt.show()

    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(data.x)

    # 頂点特徴 X の可視化
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(G, ax=ax, pos=X_tsne, node_size=20, node_color=cs, labels={i: '' for i in range(n)}, edge_color='#84919e', width=0.1)

    plt.show()

    H = (data.x @ model.conv1.lin.weight.T).detach().numpy()
    tsne = TSNE(n_components=2, random_state=0)
    H_tsne = tsne.fit_transform(H)

    # GCN において集約の直前の埋め込みの可視化
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx(G, ax=ax, pos=H_tsne, node_size=20, node_color=cs, labels={i: '' for i in range(n)}, edge_color='#84919e', width=0.1)

    plt.show()

if __name__ == '__main__':
    main()