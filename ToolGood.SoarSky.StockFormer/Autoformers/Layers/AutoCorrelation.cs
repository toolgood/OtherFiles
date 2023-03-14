using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    // 
    //     AutoCorrelation Mechanism with the following two phases:
    //     (1) period-based dependencies discovery
    //     (2) time delay aggregation
    //     This block can replace the self-attention family mechanism seamlessly.
    //     
    public class AutoCorrelation : nn.Module
    {
        public bool training;
        public Dropout dropout;
        public double factor;
        public bool mask_flag;
        public bool output_attention;
        public double? scale;

        public AutoCorrelation(
            bool mask_flag = true,
            int factor = 1,
            double? scale = null,
            double attention_dropout = 0.1,
            bool output_attention = false) : base("AutoCorrelation")
        {
            this.factor = factor;
            this.scale = scale;
            this.mask_flag = mask_flag;
            this.output_attention = output_attention;
            this.dropout = nn.Dropout(attention_dropout);
            this.RegisterComponents();

        }

        // 
        //         SpeedUp version of Autocorrelation (a batch-normalization style design)
        //         This is for the training phase.
        //         
        public virtual Tensor time_delay_agg_training(Tensor values, Tensor corr)
        {
            var head = values.shape[1];
            var channel = values.shape[2];
            var length = values.shape[3];
            // find top k
            var top_k = Convert.ToInt32(this.factor * Math.Log(length));
            var mean_value = torch.mean(torch.mean(corr, dimensions: new long[] { 1 }), dimensions: new long[] { 1 });
            var index = torch.topk(torch.mean(mean_value, dimensions: new long[] { 0 }), top_k, dim: -1).indices;
            var weights = torch.stack((from i in Enumerable.Range(0, top_k)
                                       select mean_value[TensorIndex.Colon, index[i]]).ToList(), dim: -1);
            // update corr
            var tmp_corr = torch.softmax(weights, dim: -1);
            // aggregation
            var tmp_values = values;
            var delays_agg = torch.zeros_like(values).@float();
            foreach (var i in Enumerable.Range(0, top_k)) {
                var pattern = torch.roll(tmp_values, -Convert.ToInt32(index[i]), -1);
                delays_agg = delays_agg + pattern * tmp_corr[TensorIndex.Colon, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length);
            }
            return delays_agg;
        }

        // 
        //         SpeedUp version of Autocorrelation (a batch-normalization style design)
        //         This is for the inference phase.
        //         
        public virtual Tensor time_delay_agg_inference(Tensor values, Tensor corr)
        {
            var batch = values.shape[0];
            var head = values.shape[1];
            var channel = values.shape[2];
            var length = values.shape[3];
            // index init
            var init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device);
            // find top k
            var top_k = Convert.ToInt32(this.factor * Math.Log(length));
            var mean_value = torch.mean(torch.mean(corr, dimensions: new long[] { 1 }), dimensions: new long[] { 1 });
            var (weights, delay) = torch.topk(mean_value, top_k, dim: -1);
            // update corr
            var tmp_corr = torch.softmax(weights, dim: -1);
            // aggregation
            var tmp_values = values.repeat(1, 1, 1, 2);
            var delays_agg = torch.zeros_like(values).@float();
            foreach (var i in Enumerable.Range(0, top_k)) {
                var tmp_delay = init_index + delay[TensorIndex.Colon, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length);
                var pattern = torch.gather(tmp_values, dim: -1, index: tmp_delay);
                delays_agg = delays_agg + pattern * tmp_corr[TensorIndex.Colon, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length);
            }
            return delays_agg;
        }

        // 
        //         Standard version of Autocorrelation
        //         
        public virtual Tensor time_delay_agg_full(Tensor values, Tensor corr)
        {
            var batch = values.shape[0];
            var head = values.shape[1];
            var channel = values.shape[2];
            var length = values.shape[3];
            // index init
            var init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device);
            // find top k
            var top_k = Convert.ToInt32(this.factor * Math.Log(length));
            var (weights, delay) = torch.topk(corr, top_k, dim: -1);
            // update corr
            var tmp_corr = torch.softmax(weights, dim: -1);
            // aggregation
            var tmp_values = values.repeat(1, 1, 1, 2);
            var delays_agg = torch.zeros_like(values).@float();
            foreach (var i in Enumerable.Range(0, top_k)) {
                var tmp_delay = init_index + delay[default, i].unsqueeze(-1);
                var pattern = torch.gather(tmp_values, dim: -1, index: tmp_delay);
                delays_agg = delays_agg + pattern * tmp_corr[default, i].unsqueeze(-1);
            }
            return delays_agg;
        }

        public virtual (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor values, IMasking attn_mask)
        {
            Tensor V;
            var (B, L, H, E) = queries.shape.ToLong4();
            var (_, S, _, D) = values.shape.ToLong4();
            if (L > S) {
                var zeros = torch.zeros_like(queries[TensorIndex.Colon, TensorIndex.Slice(null, (L - S), null), TensorIndex.Colon]).@float();
                values = torch.cat(new List<Tensor> { values, zeros }, dim: 1);
                keys = torch.cat(new List<Tensor> { keys, zeros }, dim: 1);
            } else {
                values = values[TensorIndex.Colon, TensorIndex.Slice(null, L, null), TensorIndex.Colon, TensorIndex.Colon];
                keys = keys[TensorIndex.Colon, TensorIndex.Slice(null, L, null), TensorIndex.Colon, TensorIndex.Colon];
            }
            // period-based dependencies
            var q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim: -1);
            var k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim: -1);
            var res = q_fft * torch.conj(k_fft);
            var corr = torch.fft.irfft(res, dim: -1);
            // time delay agg
            if (this.training) {
                V = this.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2);
            } else {
                V = this.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2);
            }
            if (this.output_attention) {
                return (V.contiguous(), corr.permute(0, 3, 1, 2));
            } else {
                return (V.contiguous(), null);
            }
        }
    }

}
