'''
Attenion implementation
Adapted from torchnlp:
    https://github.com/PetrochukM/PyTorch-NLP/tree/master/torchnlp/nn
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
try:
    from .layers import get_activation, get_normalization_2d
except:
    from layers import get_activation, get_normalization_2d


class AttnBlock(nn.Module):
    def __init__(self,
                 attn_dim_in,
                 attn_dim_out,
                 attn_num_query=16,
                 attn_type='general',
                 normalization='none'):
        super(AttnBlock, self).__init__()
        self.attn_layer = AttnLayer(
            attn_dim_in,
            attn_dim_out,
            attn_num_query,
            attn_type,
            normalization)

    def forward(self, cat_tensor, context):
        N, C, H, W = cat_tensor.shape
        attn_map = self.attn_layer(context.transpose(1, 2))
        #print('attn_map 1:', attn_map.shape)
        attn_map = F.interpolate(attn_map, (H, W))
        #print('attn_map 2:', attn_map.shape)
        map_out = torch.cat([cat_tensor, attn_map], 1)

        return map_out

    def return_attn_weights(self):
        return self.attn_layer.attn_weights


class AttnLayer(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_query=16,
                 attention_type='general',
                 normalization='none'):
        super(AttnLayer, self).__init__()
        self.num_query = num_query
        self.query = nn.Parameter(
            torch.randn((1, num_query, dim_in),
            requires_grad=True))
        self.attention = Attention(
            dimensions=dim_in,
            dim_out=dim_out,
            attention_type=attention_type)
        self.norm_layer = get_normalization_2d(
            dim_out, normalization)
        # Cache Attenion Weights for visualization
        self.attn_weights = None

    def forward(self, context):
        N, L, D_in = context.shape
        # Repeat query layer over batch dim
        query = self.query.repeat(N, 1, 1)
        # **output** (:class:`torch.FloatTensor`
        # [batch size, output length, dim_out]
        attn_out, attn_weights = self.attention(query, context)
        self.attn_weights = attn_weights
        N, Q_L, D_out = attn_out.shape
        # [batch size, dim_out, output length]
        attn_out = torch.transpose(attn_out, 1, 2)
        #  [batch size, dimensions, H, W]
        H = W = int(sqrt(self.num_query))
        attn_map = attn_out.view(
            N, D_out, H, W)
        if self.norm_layer is not None:
            attn_map = self.norm_layer(attn_map)

        return attn_map


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self,
                 dimensions,
                 dim_out,
                 attention_type='general',
                 linear_out_layer=True,
                 ignore_tanh=False):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        # Attention Setup
        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # Output Setup
        self.linear_out_layer = linear_out_layer
        if self.linear_out_layer:
            self.dim_out = dim_out
            self.linear_out = nn.Linear(dimensions * 2, dim_out, bias=False)

        self.ignore_tanh = ignore_tanh
        if not ignore_tanh:
            self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length,
                dimensions]): Sequence of queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length,
                dimensions]): Data overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length,
                dimensions]):
                    Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size,
                output length, query length]):
                    Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * \
        # (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        # batch matrix-matrix product
        attention_scores = torch.bmm(
            query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(
            batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(
            batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * \
        # (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        if self.linear_out_layer:
            # Concatenate the attention output with the context and process the
            # concatenated feature with a fc output layer
            # concat -> (batch_size * output_len, 2*dimensions)
            combined = torch.cat((mix, query), dim=2)
            combined = combined.view(batch_size * output_len, 2 * dimensions)
            # Apply linear_out on every 2nd dimension of concat
            # output -> (batch_size, output_len, dimensions)
            output = self.linear_out(combined).view(
                batch_size, output_len, self.dim_out)
        else:
            # Directly output the attention output
            output = mix

        if not self.ignore_tanh:
            output = self.tanh(output)

        return output, attention_weights


if __name__ == '__main__':
    attention = Attention(256, 100, linear_out_layer=False)
    query = torch.randn(5, 1, 256)
    context = torch.randn(5, 5, 256)
    cat_tensor = torch.randn(5, 170, 128, 128)
    output, weights = attention(query, context)
    print('output.shape:', output.shape)
    '''
    attn_layer = AttnLayer(256, 110, 16)
    attn_map = attn_layer(context)
    print('attn_map.shape:', attn_map.shape)
    for param in attn_layer.parameters():
        print(type(param.data), param.size())

    attn_block = AttnBlock(256, 110, 16, normalization='batch')
    cat_map = attn_block(cat_tensor, context.transpose(1, 2))
    print('cat_map.shape:', cat_map.shape)
    '''
