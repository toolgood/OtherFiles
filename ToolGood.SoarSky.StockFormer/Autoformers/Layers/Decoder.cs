using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    // 
    //     Autoformer encoder
    //     
    public class Decoder : nn.Module
    {
        public ModuleList<DecoderLayer> layers;
        public my_Layernorm norm;
        public Module<Tensor, Tensor> projection;

        public Decoder(DecoderLayer[] layers, my_Layernorm norm_layer = null, Module<Tensor, Tensor> projection = null) : base("Decoder")
        {
            this.layers = nn.ModuleList(layers);
            this.norm = norm_layer;
            this.projection = projection;
            this.RegisterComponents();

        }

        public virtual (Tensor, Tensor) forward(
            Tensor x,
            Tensor cross,
            IMasking x_mask = null,
            IMasking cross_mask = null,
            Tensor trend = null)
        {
            foreach (var layer in this.layers) {
                (x, var residual_trend) = layer.forward(x, cross, x_mask: x_mask, cross_mask: cross_mask);
                trend = trend + residual_trend;
            }
            if (this.norm is not null) {
                x = this.norm.forward(x);
            }
            if (this.projection is not null) {
                x = this.projection.forward(x);
            }
            return (x, trend);
        }
    }
}
