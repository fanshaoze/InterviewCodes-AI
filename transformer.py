import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out


# Example usage
embed_size = 256
heads = 8
values = torch.rand(64, 10, embed_size)  # Batch size of 64, sequence length of 10
keys = torch.rand(64, 10, embed_size)
queries = torch.rand(64, 10, embed_size)
mask = None  # You can add an actual mask if needed

attention_layer = MultiHeadSelfAttention(embed_size, heads)
out = attention_layer(values, keys, queries, mask)
print(out.shape)  # Should be (64, 10, 256)


"""
class MHA(nn.module):

    def __init__(self, emb_size, head):
        super(MHA, self).__init__()
        self.emb_size = emb_size
        self.head = head
        assert self.emb_size % self.head == 0
        self.head_dim = self.emb_size // self.head
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q_f = nn.Linear(self.emb_size, self.self.emb_size)
        self.k_f = nn.Linear(self.emb_size, self.self.emb_size)
        self.v_f = nn.Linear(self.emb_size, self.self.emb_size)

        self.softmax = nn.Softmax(self.head)

        self.output = nn.Linear(self.emb_size, self.self.emb_size)

    def forward(self, q, k, v, mask):
        batch_size, dim, _ = q
        q = self.q_f(q)
        q = q.view(batch_size, self.head, self.head_dim)
        k = self.k_f(k)
        k = k.view(batch_size, self.head, self.head_dim)
        v = self.v_f(v)
        v = v.view(batch_size, self.head, self.head_dim)
        scores = torch.einsum("bihd, bjhd->bijh", q, k)

        scores = nn.functional.dropout(scores)

        scores *= self.scale

        if mask:
            scores = scores.masked_fill(mask == 0, 1e-9)
        attn = self.softmax(scores)

        x = torch.einsum("bijh, bihd->bijd", attn, v)
        x = x.view(batch_size, dim, -1)
        x = self.output(x)
        return x


class transformer(nn.Module):
    def __int__(self, emb_size, head):
        super(transformer, self).__init__()
        self.MHA_0 = MHA(emb_size, head)
        self.MHA_1 = MHA(emb_size, head)
        self.ff1, self.ff2 = nn.Linear(emb_size, emb_size * 4), nn.Linear(emb_size * 4, emb_size)
        self.ff3, self.ff4 = nn.Linear(emb_size, emb_size * 4), nn.Linear(emb_size * 4, emb_size)
        self.pred = nn.Linear(emb_size, 1)

    def forward(self, x):
        x_f = self.MHA_0(x, x, x, None)
        x_f = nn.functional.dropout(x_f)
        x_f = nn.functional.relu(x_f)
        x = x+x_f
        x = self.ff2(nn.functional.relu(self.ff1(x)))

        x_f = self.MHA_1(x, x, x, None)
        x_f = nn.functional.dropout(x_f)
        x_f = nn.functional.relu(x_f)
        x = x+x_f
        x = self.ff4(nn.functional.relu(self.ff3(x)))
        self.pred(x)
        return x


emb_size = 256
head = 4
x = torch.rand(10, 64, emb_size)
x_gnd = torch.rand(10)
trans = transformer(emb_size, head)
x = trans(x)
opt = torch.optim.Adam(trans.parameters(), lr = 0.001)
for epoch in range(10):
    pred = x.trans(x)
    loss = nn.functional.mse_loss(pred, x_gnd)
    opt.zero_grad()
    loss.backward()
    opt.step()


"""
