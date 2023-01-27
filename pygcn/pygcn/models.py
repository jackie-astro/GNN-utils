import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        relu -> dropout -> gc2 -> softmax
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

#
# if __name__ == '__main__':
#     from pygcn.utils import load_data
#     model = GCN(nfeat=1433, nhid=16, nclass=7, dropout=0.2)
#     model = model.cuda()
#     adj, features, labels, idx_train, idx_val, idx_test = load_data()
#     adj = adj.cuda()
#     features = features.cuda()
#     labels = labels.cuda()
#     output = model(features, adj)
#     #print(model)
