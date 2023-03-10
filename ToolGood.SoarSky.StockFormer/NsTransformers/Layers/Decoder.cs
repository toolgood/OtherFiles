using static TorchSharp.torch;
using TorchSharp.Modules;
using ToolGood.SoarSky.StockFormer.Utils;

namespace ToolGood.SoarSky.StockFormer.NsTransformers.Layers
{
    public class Decoder : nn.Module
    {

        public ModuleList<DecoderLayer> layers;
        public LayerNorm norm;
        public Linear projection;

        public Decoder(DecoderLayer[] layers, LayerNorm norm_layer = null, Linear projection = null) : base("Decoder")
        {
            this.layers = nn.ModuleList(layers);
            this.norm = norm_layer;
            this.projection = projection;
        }

        public virtual Tensor forward(Tensor x, Tensor cross, IMasking x_mask = null, IMasking cross_mask = null,
                                      Tensor tau = null, Tensor delta = null)
        {
            foreach (var layer in this.layers) {
                x = layer.forward(x, cross, x_mask: x_mask, cross_mask: cross_mask, tau: tau, delta: delta);
            }
            if (this.norm is not null) {
                x = this.norm.forward(x);
            }
            if (this.projection is not null) {
                x = this.projection.forward(x);
            }
            return x;
        }
    }

}
