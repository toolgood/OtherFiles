using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;


namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class TSTEncoderLayer : Module
    {
        public Tensor attn;
        public Dropout dropout_attn;
        public Dropout dropout_ffn;
        public Sequential ff;
        public Module<Tensor, Tensor> norm_attn;
        public Module<Tensor, Tensor> norm_ffn;
        public bool pre_norm;
        public bool res_attention;
        public _MultiheadAttention self_attn;
        public bool store_attn;

        public TSTEncoderLayer(int q_len, int d_model, int n_heads, int? d_k = null, int? d_v = null, int? d_ff = 256,
                               bool store_attn = false, string norm = "BatchNorm", double attn_dropout = 0,
                               double dropout = 0.0, bool bias = true, string activation = "gelu",
                               bool res_attention = false, bool pre_norm = false) : base("TSTEncoderLayer")
        {
            //Debug.Assert(!(d_model % n_heads));
            //Debug.Assert($"d_model ({d_model}) must be divisible by n_heads ({n_heads})");
            d_k = d_k is null ? d_model / n_heads : d_k;
            d_v = d_v is null ? d_model / n_heads : d_v;
            // Multi-Head attention
            this.res_attention = res_attention;
            self_attn = new _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout: attn_dropout, proj_dropout: dropout, res_attention: res_attention);
            // Add & Norm
            dropout_attn = Dropout(dropout);
            if (norm.lower().Contains("batch")) {
                norm_attn = Sequential(new Transpose(1, 2), BatchNorm1d(d_model), new Transpose(1, 2));
            } else {
                norm_attn = LayerNorm(d_model);
            }
            // Position-wise Feed-Forward
            ff = Sequential(Linear(d_model, d_ff.Value, hasBias: bias), PatchTST_layers.get_activation_fn(activation), Dropout(dropout), Linear(d_ff.Value, d_model, hasBias: bias));
            // Add & Norm
            dropout_ffn = Dropout(dropout);
            if (norm.lower().Contains("batch")) {
                norm_ffn = Sequential(new Transpose(1, 2), BatchNorm1d(d_model), new Transpose(1, 2));
            } else {
                norm_ffn = LayerNorm(d_model);
            }
            this.pre_norm = pre_norm;
            this.store_attn = store_attn;
        }

        public virtual (Tensor, Tensor) forward(Tensor src, Tensor prev = null, Tensor key_padding_mask = null, Tensor attn_mask = null)
        {
            // Multi-Head attention sublayer
            if (pre_norm) {
                src = norm_attn.forward(src);
            }
            //# Multi-Head attention
            Tensor attn, src2, scores = null;
            if (res_attention) {
                (src2, attn, scores) = self_attn.forward(src, src, src, prev, key_padding_mask: key_padding_mask, attn_mask: attn_mask);
            } else {
                (src2, attn, _) = self_attn.forward(src, src, src, key_padding_mask: key_padding_mask, attn_mask: attn_mask);
            }
            if (store_attn) {
                this.attn = attn;
            }
            //# Add & Norm
            src = src + dropout_attn.forward(src2);
            if (!pre_norm) {
                src = norm_attn.forward(src);
            }
            // Feed-forward sublayer
            if (pre_norm) {
                src = norm_ffn.forward(src);
            }
            //# Position-wise Feed-Forward
            src2 = ff.forward(src);
            //# Add & Norm
            src = src + dropout_ffn.forward(src2);
            if (!pre_norm) {
                src = norm_ffn.forward(src);
            }
            if (res_attention) {
                return (src, scores);
            } else {
                return (src, null);
            }
        }
    }


}
