using static TorchSharp.torch;
using TorchSharp;
using System.Reflection;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public static class PatchTST_layers
    {
        public static Module<Tensor, Tensor> get_activation_fn(string activation)
        {
            if (activation.lower() == "relu") {
                return ReLU();
            } else if (activation.lower() == "gelu") {
                return GELU();
            }
            throw new Exception($"{activation} is not available. You can use \"relu\", \"gelu\", or a callable");
        }
        public static Tensor PositionalEncoding(long q_len, long d_model, bool normalize = true)
        {
            var pe = zeros(q_len, d_model);
            var position = arange(0, q_len).unsqueeze(1);
            var div_term = exp(arange(0, d_model, 2) * -Math.Log(10000.0) / d_model);
            pe[TensorIndex.Colon, TensorIndex.Slice(0, null, 2)] = sin(position * div_term);
            pe[TensorIndex.Colon, TensorIndex.Slice(1, null, 2)] = cos(position * div_term);
            if (normalize) {
                pe = pe - pe.mean();
                pe = pe / (pe.std() * 10);
            }
            return pe;
        }

        public static Tensor Coord2dPosEncoding(
          long q_len,
          long d_model,
          bool exponential = false,
          bool normalize = true,
          double eps = 0.001,
          bool verbose = false)
        {
            Tensor cpe = null;
            var x = exponential ? 0.5 : 1;
            for (int i = 0; i < 100; i++) {
                cpe = 2 * pow(linspace(0, 1, q_len).reshape(new long[] { -1, 1 }), x) * pow(linspace(0, 1, d_model).reshape(new long[] { 1, -1 }), x) - 1;
                //pv($"{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}", verbose);
                if (abs(cpe.mean()).item<double>() <= eps) {
                    break;
                } else if (cpe.mean().item<double>() > eps) {
                    x += 0.001;
                } else {
                    x -= 0.001;
                }
            }
            if (normalize) {
                cpe = cpe - cpe.mean();
                cpe = cpe / (cpe.std() * 10);
            }
            return cpe;
        }
        public static Tensor Coord1dPosEncoding(long q_len, bool exponential = false, bool normalize = true)
        {
            var cpe = 2 * pow(linspace(0, 1, q_len).reshape(-1, 1), exponential ? 0.5 : 1) - 1;
            if (normalize) {
                cpe = cpe - cpe.mean();
                cpe = cpe / (cpe.std() * 10);
            }
            return cpe;
        }
        public static Tensor positional_encoding(string pe, bool learn_pe, int q_len, int d_model)
        {
            Tensor W_pos;
            // Positional encoding
            if (pe == null) {
                W_pos = empty(new long[] { q_len, d_model });
                init.uniform_(W_pos, -0.02, 0.02);
                learn_pe = false;
            } else if (pe == "zero") {
                W_pos = empty(new long[] { q_len, 1 });
                init.uniform_(W_pos, -0.02, 0.02);
            } else if (pe == "zeros") {
                W_pos = empty(new long[] { q_len, d_model });
                init.uniform_(W_pos, -0.02, 0.02);
            } else if (pe == "normal" || pe == "gauss") {
                W_pos = zeros(new long[] { q_len, 1 });
                init.normal_(W_pos, mean: 0.0, std: 0.1);
            } else if (pe == "uniform") {
                W_pos = zeros(new long[] { q_len, 1 });
                init.uniform_(W_pos, 0.0, 0.1);
            } else if (pe == "lin1d") {
                W_pos = Coord1dPosEncoding(q_len, exponential: false, normalize: true);
            } else if (pe == "exp1d") {
                W_pos = Coord1dPosEncoding(q_len, exponential: true, normalize: true);
            } else if (pe == "lin2d") {
                W_pos = Coord2dPosEncoding(q_len, d_model, exponential: false, normalize: true);
            } else if (pe == "exp2d") {
                W_pos = Coord2dPosEncoding(q_len, d_model, exponential: true, normalize: true);
            } else if (pe == "sincos") {
                W_pos = PositionalEncoding(q_len, d_model, normalize: true);
            } else {
                throw new Exception($"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal','zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)");
            }
            return Parameter(W_pos, requires_grad: learn_pe);
        }

    }

}
