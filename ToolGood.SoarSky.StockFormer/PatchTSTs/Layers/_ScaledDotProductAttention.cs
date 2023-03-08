using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;
using NumpyDotNet;
using F = TorchSharp.torch.nn.functional;


namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class _ScaledDotProductAttention : nn.Module
    {
        public Dropout attn_dropout;
        public bool lsa;
        public bool res_attention;
        public Parameter scale;

        public _ScaledDotProductAttention(int d_model, int n_heads, double attn_dropout = 0.0,
                                          bool res_attention = false, bool lsa = false) : base("_ScaledDotProductAttention")
        {
            this.attn_dropout = nn.Dropout(attn_dropout);
            this.res_attention = res_attention;
            var head_dim = d_model / n_heads;
            scale = nn.Parameter(tensor(Math.Pow(head_dim, -0.5)), requires_grad: lsa);
            this.lsa = lsa;
        }

        // 
        //         Input shape:
        //             q               : [bs x n_heads x max_q_len x d_k]
        //             k               : [bs x n_heads x d_k x seq_len]
        //             v               : [bs x n_heads x seq_len x d_v]
        //             prev            : [bs x n_heads x q_len x seq_len]
        //             key_padding_mask: [bs x seq_len]
        //             attn_mask       : [1 x seq_len x seq_len]
        //         Output shape:
        //             output:  [bs x n_heads x q_len x d_v]
        //             attn   : [bs x n_heads x q_len x seq_len]
        //             scores : [bs x n_heads x q_len x seq_len]
        //         
        public virtual (Tensor, Tensor, Tensor) forward(Tensor q, Tensor k, Tensor v,
            Tensor prev = null, Tensor key_padding_mask = null, Tensor attn_mask = null)
        {
            // Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
            var attn_scores = matmul(q, k) * scale;
            // Add pre-softmax attention scores from the previous layer (optional)
            if (prev is not null)
            {
                attn_scores = attn_scores + prev;
            }
            // Attention mask (optional)
            if (attn_mask is not null)
            {
                // attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
                if (attn_mask.dtype == @bool)
                {
                    attn_scores.masked_fill_(attn_mask, -np.Inf);
                }
                else
                {
                    attn_scores += attn_mask;
                }
            }
            // Key padding mask (optional)
            if (key_padding_mask is not null)
            {
                // mask with shape [bs x q_len] (only when max_w_len == q_len)
                attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.Inf);
            }
            // normalize the attention weights
            var attn_weights = F.softmax(attn_scores, dim: -1);
            attn_weights = attn_dropout.forward(attn_weights);
            // compute the new values given the attention weights
            var output = matmul(attn_weights, v);
            if (res_attention)
            {
                return (output, attn_weights, attn_scores);
            }
            else
            {
                return (output, attn_weights, null);
            }
        }
    }


}
