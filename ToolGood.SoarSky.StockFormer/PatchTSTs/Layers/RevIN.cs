using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;
using TorchSharp.Modules;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class RevIN : nn.Module
    {
        public bool affine;
        public Parameter affine_bias;
        public Parameter affine_weight;
        public double eps;
        public Tensor last;
        public Tensor mean;
        public int num_features;
        public Tensor stdev;
        public bool subtract_last;

        public RevIN(int num_features, double eps = 1E-05, bool affine = true, bool subtract_last = false) : base("RevIN")
        {
            this.num_features = num_features;
            this.eps = eps;
            this.affine = affine;
            this.subtract_last = subtract_last;
            if (this.affine) {
                _init_params();
            }
        }

        public virtual Tensor forward(Tensor x, string mode)
        {
            if (mode == "norm") {
                _get_statistics(x);
                x = _normalize(x);
            } else if (mode == "denorm") {
                x = _denormalize(x);
            } else {
                throw new NotImplementedException();
            }
            return x;
        }

        public virtual void _init_params()
        {
            // initialize RevIN params: (C,)
            affine_weight = nn.Parameter(ones(num_features));
            affine_bias = nn.Parameter(zeros(num_features));
        }

        public virtual void _get_statistics(Tensor x)
        {
            long[] dim2reduce = Enumerable.Range(1, (int)x.ndim - 1 - 1).Select(q => (long)q).ToArray();
            if (subtract_last) {
                var last = x.size(1) - 1;
                this.last = x[TensorIndex.Colon, last, TensorIndex.Colon].unsqueeze(1);
            } else {
                mean = torch.mean(x, dimensions: dim2reduce, keepdim: true).detach();
            }
            stdev = sqrt(var(x, dimensions: dim2reduce, keepdim: true, unbiased: false) + eps).detach();
        }

        public virtual Tensor _normalize(Tensor x)
        {
            if (subtract_last) {
                x = x - last;
            } else {
                x = x - mean;
            }
            x = x / stdev;
            if (affine) {
                x = x * affine_weight;
                x = x + affine_bias;
            }
            return x;
        }

        public virtual Tensor _denormalize(Tensor x)
        {
            if (affine) {
                x = x - affine_bias;
                x = x / (affine_weight + eps * eps);
            }
            x = x * stdev;
            if (subtract_last) {
                x = x + last;
            } else {
                x = x + mean;
            }
            return x;
        }
    }


}
