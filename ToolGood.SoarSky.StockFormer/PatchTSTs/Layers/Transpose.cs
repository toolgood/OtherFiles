using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class Transpose : nn.Module<Tensor, Tensor>
    {
        public bool contiguous;
        public long[] dims;

        public Transpose(bool contiguous = false, params long[] dims) : base("Transpose")
        {
            this.dims = dims;
            this.contiguous = contiguous;
        }
        public Transpose(params long[] dims) : base("Transpose")
        {
            this.dims = dims;
            contiguous = false;
        }


        public override Tensor forward(Tensor x)
        {
            if (contiguous)
            {
                return x.transpose(dims[0], dims[1]).contiguous();
            }
            else
            {
                return x.transpose(dims[0], dims[1]);
            }
        }
    }
}
