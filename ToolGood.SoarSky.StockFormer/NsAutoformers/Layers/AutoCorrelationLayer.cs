using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Layers
{
    public class AutoCorrelationLayer : nn.Module
    {
        public DSAutoCorrelation inner_correlation;
        public Linear key_projection;
        public int n_heads;
        public Linear out_projection;
        public Linear query_projection;
        public Linear value_projection;
        public AutoCorrelationLayer(
            DSAutoCorrelation correlation,
            int d_model, int n_heads,
            int? d_keys = null, int? d_values = null) : base("AutoCorrelationLayer")
        {
            d_keys = d_keys ?? d_model / n_heads;
            d_values = d_values ?? d_model / n_heads;
            inner_correlation = correlation;
            query_projection = nn.Linear(d_model, d_keys.Value * n_heads);
            key_projection = nn.Linear(d_model, d_keys.Value * n_heads);
            value_projection = nn.Linear(d_model, d_values.Value * n_heads);
            out_projection = nn.Linear(d_values.Value * n_heads, d_model);
            this.n_heads = n_heads;

            register_module("inner_correlation", inner_correlation);
            register_module("query_projection", query_projection);
            register_module("key_projection", key_projection);
            register_module("value_projection", value_projection);
            register_module("out_projection", out_projection);

            RegisterComponents();
        }

        public virtual (Tensor, Tensor) forward(
            Tensor queries, Tensor keys, Tensor values, IMasking attn_mask,
            Tensor tau = null, Tensor delta = null)
        {
            var (B, L, _) = queries.shape.ToLong3();
            var (_, S, _) = keys.shape.ToLong3();
            var H = n_heads;
            queries = query_projection.forward(queries).view(B, L, H, -1);
            keys = key_projection.forward(keys).view(B, S, H, -1);
            values = value_projection.forward(values).view(B, S, H, -1);
            var (@out, attn) = inner_correlation.forward(queries, keys, values, attn_mask, tau, delta);
            @out = @out.view(B, L, -1);
            return (out_projection.forward(@out), attn);
        }
    }
}
