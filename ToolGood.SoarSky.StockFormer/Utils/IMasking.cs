using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Utils
{
    public interface IMasking
    {
        Tensor mask { get; }
    }

    public class TriangularCausalMask : IMasking
    {

        public Tensor _mask;

        public TriangularCausalMask(long B, long L, string device = "cpu")
        {
            var mask_shape = new long[] { B, 1, L, L };
            using (var _no_grad = no_grad())
            {
                _mask = triu(ones(mask_shape, dtype: @bool), diagonal: 1).to(device);
            }
        }
        public TriangularCausalMask(long B, long L, Device device)
        {
            var mask_shape = new long[] { B, 1, L, L };
            using (var _no_grad = no_grad())
            {
                _mask = triu(ones(mask_shape, dtype: @bool), diagonal: 1).to(device);
            }
        }

        public Tensor mask
        {
            get
            {
                return _mask;
            }
        }
    }

    public class ProbMask : IMasking
    {

        public Tensor _mask;

        public ProbMask(long B, long H, long L, Tensor index, Tensor scores, string device = "cpu")
        {
            var _mask = ones(L, scores.shape[^1], dtype: @bool).to(device).triu(1);
            var _mask_ex = _mask[null, null, TensorIndex.Colon].expand(B, H, L, scores.shape[^1]);
            var indicator = _mask_ex[arange(B)[TensorIndex.Colon, null, null], arange(H)[null, TensorIndex.Colon, null], index, TensorIndex.Colon].to(device);
            this._mask = indicator.view(scores.shape).to(device);
        }
        public ProbMask(long B, long H, long L, Tensor index, Tensor scores, Device device)
        {
            var _mask = ones(L, scores.shape[^1], dtype: @bool).to(device).triu(1);
            var _mask_ex = _mask[TensorIndex.Null, TensorIndex.Null, TensorIndex.Colon].expand(B, H, L, scores.shape[^1]);
            var indicator = _mask_ex[TensorIndex.Tensor(arange(B)[TensorIndex.Colon, TensorIndex.Null, TensorIndex.Null]),
                                        TensorIndex.Tensor(arange(H)[TensorIndex.Null, TensorIndex.Colon, TensorIndex.Null]),
                                        TensorIndex.Tensor(index),
                                        TensorIndex.Colon].to(device);
            this._mask = indicator.view(scores.shape).to(device);
        }

        public Tensor mask
        {
            get
            {
                return _mask;
            }
        }
    }




}
