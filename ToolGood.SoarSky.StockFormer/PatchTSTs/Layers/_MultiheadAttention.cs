using static TorchSharp.torch;
using TorchSharp.Modules;


namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class _MultiheadAttention : nn.Module
    {
        public int d_v;
        public int d_k;
        public int n_heads;
        public bool res_attention;
        public _ScaledDotProductAttention sdp_attn;
        public Sequential to_out;
        public Linear W_K;
        public Linear W_Q;
        public Linear W_V;

        public _MultiheadAttention(int d_model, int n_heads, int? d_k = null, int? d_v = null,
                                   bool res_attention = false, double attn_dropout = 0.0, double proj_dropout = 0.0,
                                   bool qkv_bias = true, bool lsa = false) : base("_MultiheadAttention")
        {
            d_k = d_k is null ? d_model / n_heads : d_k;
            d_v = d_v is null ? d_model / n_heads : d_v;
            this.n_heads = n_heads;
            this.d_k = d_k.Value;
            this.d_v = d_v.Value;
            W_Q = nn.Linear(d_model, d_k.Value * n_heads, hasBias: qkv_bias);
            W_K = nn.Linear(d_model, d_k.Value * n_heads, hasBias: qkv_bias);
            W_V = nn.Linear(d_model, d_v.Value * n_heads, hasBias: qkv_bias);
            // Scaled Dot-Product Attention (multiple heads)
            this.res_attention = res_attention;
            sdp_attn = new _ScaledDotProductAttention(d_model, n_heads, attn_dropout: attn_dropout, res_attention: this.res_attention, lsa: lsa);
            // Poject output
            to_out = nn.Sequential(nn.Linear(n_heads * d_v.Value, d_model), nn.Dropout(proj_dropout));
        }

        public virtual (Tensor, Tensor, Tensor) forward(Tensor Q, Tensor K = null, Tensor V = null, Tensor prev = null,
                                                        Tensor key_padding_mask = null, Tensor attn_mask = null)
        {
            var bs = Q.size(0);
            if (K is null) { K = Q; }
            if (V is null) { V = Q; }

            // Linear (+ split in multiple heads)
            var q_s = W_Q.forward(Q).view(bs, -1, n_heads, d_k).transpose(1, 2);
            var k_s = W_K.forward(K).view(bs, -1, n_heads, d_k).permute(0, 2, 3, 1);
            var v_s = W_V.forward(V).view(bs, -1, n_heads, d_v).transpose(1, 2);
            // Apply Scaled Dot-Product Attention (multiple heads)
            Tensor output, attn_weights, attn_scores = null;
            if (res_attention)
            {
                (output, attn_weights, attn_scores) = sdp_attn.forward(q_s, k_s, v_s, prev: prev, key_padding_mask: key_padding_mask, attn_mask: attn_mask);
            }
            else
            {
                (output, attn_weights, _) = sdp_attn.forward(q_s, k_s, v_s, key_padding_mask: key_padding_mask, attn_mask: attn_mask);
            }
            // output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]
            // back to the original inputs dimensions
            output = output.transpose(1, 2).contiguous().view(bs, -1, n_heads * d_v);
            output = to_out.forward(output);
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
