using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Layers
{
    public class Decoder : nn.Module
    {
        public ModuleList<nn.Module> layers;
        public my_Layernorm norm;
        public nn.Module<Tensor, Tensor> projection;

        public Decoder(DecoderLayer[] layers, my_Layernorm norm_layer = null, nn.Module<Tensor, Tensor> projection = null) : base("Decoder")
        {
            this.layers = nn.ModuleList<nn.Module>(layers);
            norm = norm_layer;
            this.projection = projection;

            //for (int i = 0; i < layers.Length; i++)
            //{
            //    register_module("layers." + i, layers[i]);
            //}
            //register_module("norm", norm);
            //register_module("projection", this.projection);

            RegisterComponents();
        }

        public virtual (Tensor, Tensor) forward(
            Tensor x, Tensor cross,
            IMasking x_mask = null, IMasking cross_mask = null,
            Tensor trend = null, Tensor tau = null, Tensor delta = null)
        {
            foreach (var layer in layers)
            {
                (x, var residual_trend) = ((DecoderLayer)layer).forward(x, cross, x_mask: x_mask, cross_mask: cross_mask, tau: tau, delta: delta);
                trend = trend + residual_trend;
            }
            if (norm is not null)
            {
                x = norm.forward(x);
            }
            if (projection is not null)
            {
                x = projection.forward(x);
            }
            return (x, trend);
        }
    }
}
