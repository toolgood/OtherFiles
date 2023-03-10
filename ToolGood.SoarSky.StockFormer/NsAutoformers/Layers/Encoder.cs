using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Layers
{
    public class Encoder : nn.Module
    {
        public ModuleList<nn.Module> attn_layers;
        public ModuleList<nn.Module> conv_layers;
        public my_Layernorm norm;

        public Encoder(EncoderLayer[] attn_layers, my_Layernorm[] conv_layers = null, my_Layernorm norm_layer = null) : base("Encoder")
        {
            this.attn_layers = nn.ModuleList<nn.Module>(attn_layers);
            this.conv_layers = conv_layers is not null ? nn.ModuleList<nn.Module>(conv_layers) : null;
            norm = norm_layer;

            //for (int i = 0; i < attn_layers.Length; i++)
            //{
            //    register_module("attn_layers." + i, attn_layers[i]);
            //}
            //if (conv_layers is not null)
            //{
            //    for (int i = 0; i < conv_layers.Length; i++)
            //    {
            //        register_module("conv_layers." + i, conv_layers[i]);
            //    }
            //}
            //register_module("norm", norm);

            RegisterComponents();
        }

        public virtual (Tensor, List<Tensor>) forward(Tensor x, IMasking attn_mask = null, Tensor tau = null, Tensor delta = null)
        {
            var attns = new List<Tensor>();
            Tensor attn;
            if (conv_layers is not null)
            {
                foreach (var (attn_layer, conv_layer) in TorchEnumerable.zip(attn_layers, conv_layers))
                {
                    (x, attn) = ((EncoderLayer)attn_layer).forward(x, attn_mask: attn_mask, tau: tau, delta: delta);
                    x = ((my_Layernorm)conv_layer).forward(x);
                    attns.Add(attn);
                }
                (x, attn) = ((EncoderLayer)attn_layers[^1]).forward(x, tau: tau, delta: delta);
                attns.Add(attn);
            }
            else
            {
                foreach (var attn_layer in attn_layers)
                {
                    (x, attn) = ((EncoderLayer)attn_layer).forward(x, attn_mask: attn_mask, tau: tau, delta: delta);
                    attns.Add(attn);
                }
            }
            if (norm is not null)
            {
                x = norm.forward(x);
            }
            return (x, attns);
        }
    }
}
