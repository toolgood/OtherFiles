using CsvHelper;
using System.Diagnostics;
using System.Globalization;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace ToolGood.SoarSky.StockFormer.DataProvider
{
    public class Dataset_ETT_minute : Dataset
    {
        public string data_path;
        public List<float[]> data_stamp;
        public List<float[]> data_x;
        public List<float[]> data_y;
        public string features;
        public string freq;
        public int label_len;
        public int pred_len;
        public string root_path;
        public bool scale;
        public int seq_len;
        public int set_type;
        public string target;
        public int timeenc;

        public Dataset_ETT_minute(
            string root_path,
            string flag = "train",
            int[] size = null,
            string features = "S",
            string data_path = "ETTm1.csv",
            string target = "OT",
            bool scale = true,
            int timeenc = 0,
            string freq = "t")
        {
            // size [seq_len, label_len, pred_len]
            // info
            if (size == null) {
                this.seq_len = 24 * 4 * 4;
                this.label_len = 24 * 4;
                this.pred_len = 24 * 4;
            } else {
                this.seq_len = size[0];
                this.label_len = size[1];
                this.pred_len = size[2];
            }
            // init
            Debug.Assert(new List<string> { "train", "test", "val" }.Contains(flag));
            var type_map = new Dictionary<string, int> { { "train", 0 }, { "val", 1 }, { "test", 2 } };
            this.set_type = type_map[flag];
            this.features = features;
            this.target = target;
            this.scale = scale;
            this.timeenc = timeenc;
            this.freq = freq;
            this.root_path = root_path;
            this.data_path = data_path;
            this.@__read_data__();
        }

        public virtual void @__read_data__()
        {
            var border1s = new List<int> {
                    0,
                    12 * 30 * 24 * 4 - this.seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - this.seq_len
                };
            var border2s = new List<int> {
                    12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
                };
            var border1 = border1s[this.set_type];
            var border2 = border2s[this.set_type];


            var file = os.path.join(this.root_path, this.data_path);
            using (var reader = new StreamReader(file))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
                var records = csv.GetRecords<ETT>();

                records = records.Skip(border1).Take(border2 - border1).ToList();
                //if (this.scale) {
                //    records = records.Skip(border1s[0]).Take(border2s[0]- border1s[0]).ToList();
                //}  
                List<float[]> data = new List<float[]>();
                foreach (var item in records) {
                    float[] datas = new float[7];
                    datas[0] = (float)item.HUFL;
                    datas[1] = (float)item.HULL;
                    datas[2] = (float)item.MUFL;
                    datas[3] = (float)item.MULL;
                    datas[4] = (float)item.LUFL;
                    datas[5] = (float)item.LULL;
                    datas[6] = (float)item.OT;
                    data.Add(datas);
                }
                this.data_x = data;
                this.data_y = data;

                this.size = data.Count;

                var df_stamp = records.Select(q => q.date).ToList();
                var df_stamp2 = Timefeatures.time_features(df_stamp, this.freq);
                this.data_stamp = df_stamp2;
            }
        }


        private int size;

        public override long Count {
            get
            {
                return this.size - this.seq_len - this.pred_len + 1;
            }
        }
        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var s_begin = index;
            var s_end = s_begin + this.seq_len;
            var r_begin = s_end - this.label_len;
            var r_end = r_begin + this.label_len + this.pred_len;
            //Tensor seq_x = this.data_x[TensorIndex.Slice(s_begin, s_end, null)];
            //Tensor seq_y = this.data_y[TensorIndex.Slice(r_begin, r_end, null)];
            //Tensor seq_x_mark = this.data_stamp[TensorIndex.Slice(s_begin, s_end, null)];
            //Tensor seq_y_mark = this.data_stamp[TensorIndex.Slice(r_begin, r_end, null)];
            Tensor seq_x = GetTensor(this.data_x, s_begin, s_end);// [TensorIndex.Slice(s_begin, s_end, null)];
            Tensor seq_y = GetTensor(this.data_y, r_begin, r_end);//[TensorIndex.Slice(r_begin, r_end, null)];
            Tensor seq_x_mark = GetTensor(this.data_stamp, s_begin, s_end);//[TensorIndex.Slice(s_begin, s_end, null)];
            Tensor seq_y_mark = GetTensor(this.data_stamp, r_begin, r_end);//[TensorIndex.Slice(r_begin, r_end, null)];
            return new Dictionary<string, Tensor> {
                { "batch_x",seq_x },
                { "batch_y",seq_y },
                { "batch_x_mark",seq_x_mark },
                { "batch_y_mark",seq_y_mark },
            };
        }

        public Tensor GetTensor(List<float[]> floats, int begin, int end)
        {
            List<Tensor> list = new List<Tensor>();
            for (int i = begin; i < end; i++) {
                list.Add(torch.tensor(floats[i]));
            }
            return torch.cat(list, 0);
        }
        public Tensor GetTensor(List<float[]> floats, long begin, long end)
        {
            List<Tensor> list = new List<Tensor>();
            for (int i = (int)begin; i < (int)end; i++) {
                list.Add(torch.tensor(floats[i]));
            }
            return torch.cat(list, 0).reshape(new long[] { end - begin, floats[0].Length });
        }

    }
}
