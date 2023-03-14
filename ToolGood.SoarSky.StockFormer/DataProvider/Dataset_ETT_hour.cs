using CsvHelper;
using System.Diagnostics;
using System.Globalization;
using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static ToolGood.SoarSky.StockFormer.DataProvider.Dataset_ETT_hour;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace ToolGood.SoarSky.StockFormer.DataProvider
{
    public class Dataset_ETT_hour : Dataset
    {
        public string data_path;
        public List<double[]> data_stamp;
        public List<double[]> data_x;
        public List<double[]> data_y;
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

        public Dataset_ETT_hour(
            string root_path,
            string flag = "train",
            int[] size = null,
            string features = "S",
            string data_path = "ETTh1.csv",
            string target = "OT",
            bool scale = true,
            int timeenc = 0,
            string freq = "h")
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
            var border1s = new List<int> {  0,
                                            12 * 30 * 24 - this.seq_len,
                                            12 * 30 * 24 + 4 * 30 * 24 - this.seq_len };
            var border2s = new List<int> {  12 * 30 * 24,
                                            12 * 30 * 24 + 4 * 30 * 24,
                                            12 * 30 * 24 + 8 * 30 * 24 };
            var border1 = border1s[this.set_type];
            var border2 = border2s[this.set_type];

            var file = os.path.join(this.root_path, this.data_path);
            using (var reader = new StreamReader(file))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
                var records = csv.GetRecords<ETT>().ToList();

                var records2 = new List<ETT>();
                for (int i = border1s[0]; i < border2s[0]; i++) { records2.Add(records[i]); }
                StandardScaler2 standardScaler = new StandardScaler2();
                standardScaler.fit(records2);
                // array([ 7.93774225,  2.02103866,  5.0797706 ,  0.74618588,  2.78176239,        0.78845312, 17.1282617 ])
                // array([33.78805569,  4.36853745, 30.45708257,  3.71093711,  1.04759863,        0.39719822, 84.20798753])
                var item2 = standardScaler.transform(records[0]);
                // array([-0.36312285, -0.0057598, -0.63071223, -0.14752332, 1.38857471,        0.87514257, 1.46055158])

                List<double[]> data = new List<double[]>();
                for (int i = border1; i < border2; i++) {
                    var item = standardScaler.transform(records[i]);
                    double[] datas = new double[7];
                    datas[0] = (double)item.HUFL;
                    datas[1] = (double)item.HULL;
                    datas[2] = (double)item.MUFL;
                    datas[3] = (double)item.MULL;
                    datas[4] = (double)item.LUFL;
                    datas[5] = (double)item.LULL;
                    datas[6] = (double)item.OT;

                    data.Add(datas);
                }

                //// test StandardScaler2 data 
                //List<string> dateStr = new List<string>();
                //for (int i = 0; i < records.Count; i++) {
                //    var item = standardScaler.transform(records[i]);
                //    var datas = new string[7];
                //    datas[0] = item.HUFL.ToString("0.00000000").TrimEnd('0');
                //    datas[1] = item.HULL.ToString("0.00000000").TrimEnd('0');
                //    datas[2] = item.MUFL.ToString("0.00000000").TrimEnd('0');
                //    datas[3] = item.MULL.ToString("0.00000000").TrimEnd('0');
                //    datas[4] = item.LUFL.ToString("0.00000000").TrimEnd('0');
                //    datas[5] = item.LULL.ToString("0.00000000").TrimEnd('0');
                //    datas[6] = item.OT.ToString("0.00000000").TrimEnd('0');

                //    var str = string.Join(" ", datas);
                //    dateStr.Add(str);
                //}
                //File.WriteAllText("ETTh1.txt", string.Join("\r\n", dateStr));
                ////  Through comparison, it is found that the data of python is the same as that here 
                ////  look Datasets\ETT-small\ETTh1.txt   Datasets\ETT-small\ETTh1-python.txt

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

        public class StandardScaler2
        {
            private double[] mean;
            private double[] std;
            public StandardScaler2() { }


            public void fit(List<ETT> datas)
            {
                mean = new double[7];
                std = new double[7];

                var index = 0;
                var (m, s) = fit(datas.Select(q => q.HUFL).ToArray());
                mean[index] = m; std[index++] = s;
                (m, s) = fit(datas.Select(q => q.HULL).ToArray());

                mean[index] = m; std[index++] = s;
                (m, s) = fit(datas.Select(q => q.MUFL).ToArray());
                mean[index] = m; std[index++] = s;
                (m, s) = fit(datas.Select(q => q.MULL).ToArray());
                mean[index] = m; std[index++] = s;

                (m, s) = fit(datas.Select(q => q.LUFL).ToArray());
                mean[index] = m; std[index++] = s;
                (m, s) = fit(datas.Select(q => q.LULL).ToArray());
                mean[index] = m; std[index++] = s;
                (m, s) = fit(datas.Select(q => q.OT).ToArray());
                mean[index] = m; std[index++] = s;
            }

            public (double, double) fit(double[] nums)
            {
                //nums = nums.OrderBy(x => x).ToArray();
                var mean = nums.Average();
                var sum = 0.0;
                foreach (var num in nums) {
                    sum += (num - mean) * (num - mean);
                }
                var std = Math.Sqrt(sum / (nums.Length));
                return (mean, std);
            }



            public ETT transform(ETT data)
            {
                var index = 0;
                ETT newETT = new ETT();
                newETT.date = data.date;
                newETT.HUFL = transform(data.HUFL, index++);
                newETT.HULL = transform(data.HULL, index++);

                newETT.MUFL = transform(data.MUFL, index++);
                newETT.MULL = transform(data.MULL, index++);

                newETT.LUFL = transform(data.LUFL, index++);
                newETT.LULL = transform(data.LULL, index++);

                newETT.OT = transform(data.OT, index++);

                return newETT;
            }

            public double transform(double data, int index)
            {
                return (data - this.mean[index]) / this.std[index];
            }
        }


        public Tensor GetTensor(List<double[]> floats, long begin, long end)
        {
            double[,] list = new double[end - begin, floats[0].Length];
            var index = 0;
            for (int i = (int)begin; i < (int)end; i++) {
                for (int j = 0; j < floats[0].Length; j++) {
                    list[index, j] = floats[index][j];
                }
                index++;
            }
            return torch.tensor(list);
        }

    }



}
