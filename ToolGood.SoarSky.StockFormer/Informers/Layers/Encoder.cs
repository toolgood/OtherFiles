using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Informers.Layers
{
    public class Encoder : nn.Module
    {

        public ModuleList<EncoderLayer> attn_layers;

        public ModuleList<ConvLayer> conv_layers;

        public Module<Tensor,Tensor> norm;

        public Encoder(EncoderLayer[] attn_layers, ConvLayer[] conv_layers = null, Module<Tensor, Tensor> norm_layer = null) : base("Encoder")
        {
            this.attn_layers = nn.ModuleList(attn_layers);
            this.conv_layers = conv_layers is not null ? nn.ModuleList(conv_layers) : null;
            this.norm = norm_layer;
        }

        public virtual (Tensor, List<Tensor>) forward(Tensor x, IMasking attn_mask = null)
        {
            Tensor attn;
            // x [B, L, D]
            var attns = new List<Tensor>();
            if (this.conv_layers is not null) {
                foreach (var (attn_layer, conv_layer) in TorchEnumerable.zip(this.attn_layers, this.conv_layers)) {
                    (x, attn) = attn_layer.forward(x, attn_mask: attn_mask);
                    x = conv_layer.forward(x);
                    attns.append(attn);
                }
                (x, attn) = this.attn_layers[^1].forward(x, attn_mask: attn_mask);
                attns.append(attn);
            } else {
                foreach (var attn_layer in this.attn_layers) {
                    (x, attn) = attn_layer.forward(x, attn_mask: attn_mask);
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
