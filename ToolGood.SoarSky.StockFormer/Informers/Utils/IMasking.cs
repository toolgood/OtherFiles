using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Informers.Utils
{
    public interface IMasking
    {
        Tensor mask { get; }
    }

    public class TriangularCausalMask: IMasking
    {

        public Tensor _mask;

        public TriangularCausalMask(long B, long L, string device = "cpu")
        {
            var mask_shape = new long[] { B, 1, L, L };
            using (var _no_grad = torch.no_grad()) {
                this._mask = torch.triu(torch.ones(mask_shape, dtype: torch.@bool), diagonal: 1).to(device);
            }
        }
        public TriangularCausalMask(long B, long L, Device device)
        {
            var mask_shape = new long[] { B, 1, L, L };
            using (var _no_grad = torch.no_grad()) {
                this._mask = torch.triu(torch.ones(mask_shape, dtype: torch.@bool), diagonal: 1).to(device);
            }
        }

        public Tensor mask {
            get {
                return this._mask;
            }
        }
    }

    public class ProbMask: IMasking
    {

        public Tensor _mask;

        public ProbMask(long B, long H, long L, long index, Tensor scores, string device = "cpu")
        {
            var _mask = torch.ones(L, scores.shape[^1], dtype: torch.@bool).to(device).triu(1);
            var _mask_ex = _mask[null, null, TensorIndex.Ellipsis].expand(B, H, L, scores.shape[^1]);
            var indicator = _mask_ex[torch.arange(B)[TensorIndex.Ellipsis, null, null], torch.arange(H)[null, TensorIndex.Ellipsis, null], index, TensorIndex.Ellipsis].to(device);
            this._mask = indicator.view(scores.shape).to(device);
        }
        public ProbMask(long B, long H, long L, long index, Tensor scores, Device device)
        {
            var _mask = torch.ones(L, scores.shape[^1], dtype: torch.@bool).to(device).triu(1);
            var _mask_ex = _mask[null, null, TensorIndex.Ellipsis].expand(B, H, L, scores.shape[^1]);
            var indicator = _mask_ex[torch.arange(B)[TensorIndex.Ellipsis, null, null], torch.arange(H)[null, TensorIndex.Ellipsis, null], index, TensorIndex.Ellipsis].to(device);
            this._mask = indicator.view(scores.shape).to(device);
        }



        public Tensor mask {
            get {
                return this._mask;
            }
        }
    }




}
