using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Informers.Layers
{
    public class Decoder : nn.Module
    {

        public ModuleList<DecoderLayer> layers;

        public nn.Module<Tensor, Tensor> norm;

        public Decoder(DecoderLayer[] layers, nn.Module<Tensor, Tensor> norm_layer = null) : base("Decoder")
        {
            this.layers = nn.ModuleList(layers);
            this.norm = norm_layer;
        }

        public virtual Tensor forward(Tensor x, Tensor cross, IMasking x_mask = null, IMasking cross_mask = null)
        {
            foreach (var layer in this.layers) {
                x = layer.forward(x, cross, x_mask: x_mask, cross_mask: cross_mask);
            }
            if (this.norm is not null) {
                x = this.norm.forward(x);
            }
            return x;
        }
    }


}
