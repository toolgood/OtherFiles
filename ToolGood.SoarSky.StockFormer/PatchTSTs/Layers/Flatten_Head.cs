using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class Flatten_Head : nn.Module<Tensor, Tensor>
    {
        public Dropout dropout;
        public ModuleList<Dropout> dropouts;
        public Flatten flatten;
        public ModuleList<Flatten> flattens;
        public bool individual;
        public Linear linear;
        public ModuleList<Linear> linears;
        public int n_vars;

        public Flatten_Head(bool individual, int n_vars, long nf, long target_window, double head_dropout = 0) : base("Flatten_Head")
        {
            this.individual = individual;
            this.n_vars = n_vars;
            if (this.individual)
            {
                linears = nn.ModuleList<Linear>();
                dropouts = nn.ModuleList<Dropout>();
                flattens = nn.ModuleList<Flatten>();
                foreach (var i in Enumerable.Range(0, this.n_vars))
                {
                    flattens.append(nn.Flatten(startDim: -2));
                    linears.append(nn.Linear(nf, target_window));
                    dropouts.append(nn.Dropout(head_dropout));
                }
            }
            else
            {
                flatten = nn.Flatten(startDim: -2);
                linear = nn.Linear(nf, target_window);
                dropout = nn.Dropout(head_dropout);
            }
        }

        public override Tensor forward(Tensor x)
        {
            // x: [bs x nvars x d_model x patch_num]
            if (individual)
            {
                var x_out = new List<Tensor>();
                foreach (var i in Enumerable.Range(0, n_vars))
                {
                    var z = flattens[i].forward(x[TensorIndex.Ellipsis, i, TensorIndex.Ellipsis, TensorIndex.Ellipsis]);
                    z = linears[i].forward(z);
                    z = dropouts[i].forward(z);
                    x_out.append(z);
                }
                x = stack(x_out, dim: 1);
            }
            else
            {
                x = flatten.forward(x);
                x = linear.forward(x);
                x = dropout.forward(x);
            }
            return x;
        }
    }


}
