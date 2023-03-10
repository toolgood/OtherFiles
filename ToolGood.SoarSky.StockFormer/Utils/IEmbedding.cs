using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Utils
{
    public interface IEmbedding
    {
        Tensor forward(Tensor x);
    }

    public class PositionalEmbedding : Module, IEmbedding
    {
        public Tensor pe;

        public PositionalEmbedding(long d_model, int max_len = 5000) : base("PositionalEmbedding")
        {
            //Compute the positional encodings once in log space.
            var pe = zeros(max_len, d_model).@float();
            pe.requires_grad = false;
            var position = arange(0, max_len).@float().unsqueeze(1);
            var div_term = (arange(0, d_model, 2).@float() * -Math.Log(10000.0) / d_model).exp();
            pe[TensorIndex.Colon, TensorIndex.Slice(0, null, 2)] = sin(position * div_term);
            pe[TensorIndex.Colon, TensorIndex.Slice(1, null, 2)] = cos(position * div_term);
            pe = pe.unsqueeze(0);
            this.pe = pe;
            register_buffer("pe", pe);
        }

        public virtual Tensor forward(Tensor x)
        {
            //return pe[TensorIndex.Slice(null, x.shape[1]), TensorIndex.Slice()].unsqueeze(0);

            return pe[TensorIndex.Colon, TensorIndex.Slice(null, x.size(1), null)];
        }
    }

    public class TokenEmbedding : Module, IEmbedding
    {

        public Conv1d tokenConv;

        public TokenEmbedding(int c_in, int d_model) : base("TokenEmbedding")
        {
            var padding = 1;
            tokenConv = Conv1d(inputChannel: c_in, outputChannel: d_model, kernelSize: 3, padding: padding, paddingMode: PaddingModes.Circular);
            foreach (var m in modules()) {
                if (m is Conv1d conv1D) {
                    //nn.init.kaiming_normal_(conv1D.weight, mode: "fan_in", nonlinearity: "leaky_relu");
                    init.kaiming_normal_(conv1D.weight, mode: init.FanInOut.FanIn, nonlinearity: init.NonlinearityType.LeakyReLU);
                }
            }
        }

        public virtual Tensor forward(Tensor x)
        {
            x = tokenConv.forward(x.permute(0, 2, 1)).transpose(1, 2);
            return x;
        }
    }

    public class FixedEmbedding : Module<Tensor, Tensor>, IEmbedding
    {

        public Embedding emb;

        public FixedEmbedding(long c_in, long d_model) : base("FixedEmbedding")
        {
            var w = zeros(c_in, d_model).@float();
            w.requires_grad = false;
            var position = arange(0, c_in).@float().unsqueeze(1);
            var div_term = (arange(0, d_model, 2).@float() * -Math.Log(10000.0) / d_model).exp();
            w[TensorIndex.Colon, TensorIndex.Slice(0, null, 2)] = sin(position * div_term);
            w[TensorIndex.Colon, TensorIndex.Slice(1, null, 2)] = cos(position * div_term);
            emb = Embedding(c_in, d_model);
            emb.weight = Parameter(w, requires_grad: false);
        }

        public override Tensor forward(Tensor x)
        {
            return emb.forward(x).detach();
        }
    }

    public class TemporalEmbedding : Module<Tensor, Tensor>, IEmbedding
    {
        public Module<Tensor, Tensor> day_embed;
        public Module<Tensor, Tensor> hour_embed;
        public Module<Tensor, Tensor> minute_embed;
        public Module<Tensor, Tensor> month_embed;
        public Module<Tensor, Tensor> weekday_embed;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="d_model"></param>
        /// <param name="embed_type"></param>
        /// <param name="freq">时间特征编码的频率（默认为 h）。这可以设置为 s,t,h,d,b,w,m 
        /// (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly)。你也可以使用更详细的频率，如 15min 或 3h</param>
        public TemporalEmbedding(long d_model, string embed_type = "fixed", string freq = "h") : base(nameof(TemporalEmbedding))
        {
            var minute_size = 4;
            var hour_size = 24;
            var weekday_size = 7;
            var day_size = 32;
            var month_size = 13;
            if (embed_type == "fixed") {
                if (freq == "t") {
                    minute_embed = new FixedEmbedding(minute_size, d_model);
                    register_module("minute_embed", minute_embed);
                }
                hour_embed = new FixedEmbedding(hour_size, d_model);
                weekday_embed = new FixedEmbedding(weekday_size, d_model);
                day_embed = new FixedEmbedding(day_size, d_model);
                month_embed = new FixedEmbedding(month_size, d_model);


            } else {
                if (freq == "t") {
                    minute_embed = Embedding(minute_size, d_model);
                    register_module("minute_embed", minute_embed);
                }
                hour_embed = Embedding(hour_size, d_model);
                weekday_embed = Embedding(weekday_size, d_model);
                day_embed = Embedding(day_size, d_model);
                month_embed = Embedding(month_size, d_model);
            }
            register_module("hour_embed", hour_embed);
            register_module("weekday_embed", weekday_embed);
            register_module("day_embed", day_embed);
            register_module("month_embed", month_embed);

            RegisterComponents();
        }
        public override Tensor forward(Tensor x)
        {
            x = x.@long();
            var hour_x = hour_embed.forward(x[TensorIndex.Colon, TensorIndex.Colon, 3]);
            var weekday_x = weekday_embed.forward(x[TensorIndex.Colon, TensorIndex.Colon, 2]);
            var day_x = day_embed.forward(x[TensorIndex.Colon, TensorIndex.Colon, 1]);
            var month_x = month_embed.forward(x[TensorIndex.Colon, TensorIndex.Colon, 0]);
            if (minute_embed != null) {
                var minute_x = minute_embed.forward(x[TensorIndex.Colon, TensorIndex.Colon, 4]);
                return hour_x + weekday_x + day_x + month_x + minute_x;
            }
            return hour_x + weekday_x + day_x + month_x;
        }

    }

    public class TimeFeatureEmbedding : Module<Tensor, Tensor>, IEmbedding
    {

        public Linear embed;

        public TimeFeatureEmbedding(long d_model, string embed_type = "timeF", string freq = "h") : base("TimeFeatureEmbedding")
        {
            var freq_map = new Dictionary<string, int> {
                    { "h", 4},
                    { "t", 5},
                    { "s", 6},
                    { "m", 1},
                    { "a", 1},
                    { "w", 2},
                    { "d", 3},
                    { "b", 3},

                    { "l_d", 6},
                    { "l_h", 7},
                    { "l_t", 8},
                    { "l_s", 9},
            };
            var d_inp = freq_map[freq];
            embed = Linear(d_inp, d_model);
        }

        public override Tensor forward(Tensor x)
        {
            return embed.forward(x);
        }
    }




}
