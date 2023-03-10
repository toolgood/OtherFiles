using static TorchSharp.torch;
using TorchSharp.Modules;
using ToolGood.SoarSky.StockFormer.Utils;

namespace ToolGood.SoarSky.StockFormer.NsTransformers.Layers
{
    public class Encoder : nn.Module
    {
        public ModuleList<EncoderLayer> attn_layers;
        public ModuleList<ConvLayer> conv_layers;
        public LayerNorm norm;

        public Encoder(EncoderLayer[] attn_layers, ConvLayer[] conv_layers = null, LayerNorm norm_layer = null) : base("Encoder")
        {
            this.attn_layers = nn.ModuleList(attn_layers);
            this.conv_layers = conv_layers is not null ? nn.ModuleList(conv_layers) : null;
            this.norm = norm_layer;
        }

        public virtual (Tensor, List<Tensor>) forward(Tensor x, IMasking attn_mask = null, Tensor tau = null, Tensor delta = null)
        {
            Tensor attn;
            // x [B, L, D]
            var attns = new List<Tensor>();
            if (this.conv_layers is not null) {
                // The reason why we only import delta for the first attn_block of Encoder
                // is to integrate Informer into our framework, where row size of attention of Informer is changing each layer
                // and inconsistent to the sequence length of the initial input,
                // then no way to add delta to every row, so we make delta=0.0 (See our Appendix E.2)
                // 
                foreach (var (i, (attn_layer, conv_layer)) in TorchEnumerable.zip(this.attn_layers, this.conv_layers).Select((_p_1, _p_2) => Tuple.Create(_p_2, _p_1))) {
                    delta = i == 0 ? delta : null;
                    (x, attn) = attn_layer.forward(x, attn_mask: attn_mask, tau: tau, delta: delta);
                    x = conv_layer.forward(x);
                    attns.append(attn);
                }
                (x, attn) = this.attn_layers[^1].forward(x, tau: tau, delta: null);
                attns.append(attn);
            } else {
                foreach (var attn_layer in this.attn_layers) {
                    (x, attn) = attn_layer.forward(x, attn_mask: attn_mask, tau: tau, delta: delta);
                    attns.append(attn);
                }
            }
            if (this.norm is not null) {
                x = this.norm.forward(x);
            }
            return (x, attns);
        }
    }

}
