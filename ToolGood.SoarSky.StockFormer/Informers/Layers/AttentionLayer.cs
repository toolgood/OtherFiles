using System.Collections.Generic;
using ToolGood.SoarSky.StockFormer.Informers.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.ApiDef.Types;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Informers.Layers
{
    public class AttentionLayer : nn.Module
    {
        public nn.Module<Tensor, Tensor, Tensor, IMasking, (Tensor, Tensor)> inner_attention;
        public Linear key_projection;
        public bool mix;
        public int n_heads;
        public Linear out_projection;
        public Linear query_projection;
        public Linear value_projection;

        public AttentionLayer(
            nn.Module<Tensor, Tensor, Tensor, IMasking, (Tensor, Tensor)> attention,
            int d_model,
            int n_heads,
            int? d_keys = null,
            int? d_values = null,
            bool mix = false) : base("AttentionLayer")
        {
            d_keys = d_keys ?? d_model / n_heads; //d_keys || d_model / n_heads;
            d_values = d_values ?? d_model / n_heads; //d_values || d_model / n_heads;
            this.inner_attention = attention;
            this.query_projection = nn.Linear(d_model, d_keys.Value * n_heads);
            this.key_projection = nn.Linear(d_model, d_keys.Value * n_heads);
            this.value_projection = nn.Linear(d_model, d_values.Value * n_heads);
            this.out_projection = nn.Linear(d_values.Value * n_heads, d_model);
            this.n_heads = n_heads;
            this.mix = mix;
        }

        public virtual (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor values, IMasking attn_mask)
        {
            Tensor @out, attn;
            var (B, L, _) = queries.shape.ToLong3();
            var (_, S, _) = keys.shape.ToLong3();
            var H = this.n_heads;
            queries = this.query_projection.forward(queries).view(B, L, H, -1);
            keys = this.key_projection.forward(keys).view(B, S, H, -1);
            values = this.value_projection.forward(values).view(B, S, H, -1);
            (@out, attn) = this.inner_attention.forward(queries, keys, values, attn_mask);
            if (this.mix) {
                @out = @out.transpose(2, 1).contiguous();
            }
            @out = @out.view(B, L, -1);
            return (this.out_projection.forward(@out), attn);
        }
    }



}
