using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace ToolGood.SoarSky.StockFormer.NsTransformers.Layers
{
    public class Projector : nn.Module
    {
        public Sequential backbone;
        public Conv1d series_conv;

        public Projector(int enc_in, int seq_len, List<int> hidden_dims,
            int hidden_layers, int output_dim, int kernel_size = 3) : base("Projector")
        {
            var padding = 1;
            series_conv = nn.Conv1d(inputChannel: seq_len, outputChannel: 1, kernelSize: kernel_size, padding: padding, paddingMode: PaddingModes.Circular, bias: false);
            var layers = new List<nn.Module<Tensor, Tensor>> {
                    nn.Linear(2 * enc_in, hidden_dims[0]),
                    nn.ReLU()
                };
            foreach (var i in Enumerable.Range(0, hidden_layers - 1)) {
                layers.Add(nn.Linear(hidden_dims[i], hidden_dims[i + 1]));
                layers.Add(nn.ReLU());
            }
            layers.Add(nn.Linear(hidden_dims[^1], output_dim, hasBias: false));
            backbone = nn.Sequential(layers.ToArray());


            register_module("series_conv", series_conv);
            register_module("backbone", backbone);

            RegisterComponents();
        }

        public virtual Tensor forward(Tensor x, Tensor stats)
        {
            // x:     B x S x E
            // stats: B x 1 x E
            // y:     B x O
            var batch_size = x.shape[0];
            x = this.series_conv.forward(x);
            x = torch.cat(new List<Tensor> { x, stats }, dim: 1);
            x = x.view(batch_size, -1);
            var y = this.backbone.forward(x);
            return y;
        }
    }

}
