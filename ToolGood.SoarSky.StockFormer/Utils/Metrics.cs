using static TorchSharp.torch;
using TorchSharp;

namespace ToolGood.SoarSky.StockFormer.Utils
{
    public static class Metrics
    {

        public static Tensor RSE(Tensor pred, Tensor @true)
        {
            return sqrt(sum(pow(@true - pred, 2))) / sqrt(sum(pow(@true - @true.mean(), 2)));
        }

        public static Tensor CORR(Tensor pred, Tensor @true)
        {
            var u = ((@true - @true.mean(new long[] { 0 })) * (pred - pred.mean(new long[] { 0 }))).sum(0);
            var d = sqrt((pow(@true - @true.mean(new long[] { 0 }), 2) * pow(pred - pred.mean(new long[] { 0 }), 2)).sum(0));
            d += 1E-12;
            return 0.01 * (u / d).mean(new long[] { -1 });
        }

        public static Tensor MAE(Tensor pred, Tensor @true)
        {
            return mean(abs(pred - @true));
        }

        public static Tensor MSE(Tensor pred, Tensor @true)
        {
            return mean(pow(pred - @true, 2));
        }

        public static Tensor RMSE(Tensor pred, Tensor @true)
        {
            return sqrt(MSE(pred, @true));
        }

        public static Tensor MAPE(Tensor pred, Tensor @true)
        {
            return mean(abs((pred - @true) / @true));
        }

        public static Tensor MSPE(Tensor pred, Tensor @true)
        {
            return mean(square((pred - @true) / @true));
        }
        public static (object, object, object, object, object, object, object) metric(List<Tensor> pred, List<Tensor> @true)
        {
            var preds = cat(pred);
            var @trues = cat(@true);
            return metric(preds, @trues);
        }
        public static (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) metric(Tensor pred, Tensor @true)
        {
            var mae = MAE(pred, @true);
            var mse = MSE(pred, @true);
            var rmse = RMSE(pred, @true);
            var mape = MAPE(pred, @true);
            var mspe = MSPE(pred, @true);
            var rse = RSE(pred, @true);
            var corr = CORR(pred, @true);
            return (mae, mse, rmse, mape, mspe, rse, corr);
        }
    }

}
