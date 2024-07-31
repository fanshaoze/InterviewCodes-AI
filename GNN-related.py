import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.spmm(adj, x)
        out = self.linear(out)
        return out

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = F.relu(x)
        x = self.layer2(x, adj)
        return x

# Example usage
num_nodes = 5
in_features = 3
hidden_features = 4
out_features = 2

x = torch.rand((num_nodes, in_features))
adj = torch.eye(num_nodes)  # Identity adjacency matrix (self-loops)

model = GCN(in_features, hidden_features, out_features)
output = model(x, adj)
print(output)

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x, adj):
        neighbor_agg = torch.mm(adj, x)
        out = torch.cat([x, neighbor_agg], dim=1)
        out = self.linear(out)
        return out


class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GraphSAGE, self).__init__()
        self.layer1 = GraphSAGELayer(in_features, hidden_features)
        self.layer2 = GraphSAGELayer(hidden_features, out_features)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = F.relu(x)
        x = self.layer2(x, adj)
        return x


# Example usage
num_nodes = 5
in_features = 3
hidden_features = 4
out_features = 2

x = torch.rand((num_nodes, in_features))
adj = torch.eye(num_nodes)  # Identity adjacency matrix (self-loops)

model = GraphSAGE(in_features, hidden_features, out_features)
output = model(x, adj)
print(output)



class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features // num_heads

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.Parameter(torch.Tensor(num_heads, out_features, out_features))
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        N = x.size()[0]
        h = self.linear(x)
        h_prime = h.view(N, self.num_heads, self.out_features)

        attn_for_self = torch.matmul(h_prime, self.attention)
        attn_for_neighs = torch.matmul(attn_for_self, h_prime.transpose(0, 1))

        e = self.leakyrelu(attn_for_neighs)
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, p=0.6, training=self.training)
        h_prime = torch.matmul(attention, h_prime)

        return h_prime.view(N, -1)

class GAT(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(in_features, hidden_features, num_heads)
        self.layer2 = GATLayer(hidden_features, out_features, 1)  # Single head for output

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = F.elu(x)
        x = self.layer2(x, adj)
        return x

# Example usage
num_nodes = 5
in_features = 3
hidden_features = 8
out_features = 2
num_heads = 4

x = torch.rand((num_nodes, in_features))
adj = torch.eye(num_nodes)  # Identity adjacency matrix (self-loops)

model = GAT(in_features, hidden_features, out_features, num_heads)
output = model(x, adj)
print(output)
