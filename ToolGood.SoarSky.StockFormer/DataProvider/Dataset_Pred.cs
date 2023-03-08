//using CsvHelper;
//using System.Diagnostics;
//using System.Globalization;
//using TorchSharp;
//using static TorchSharp.torch;
//using static TorchSharp.torch.utils.data;

//namespace ToolGood.SoarSky.StockFormer.DataProvider
//{
//    public class Dataset_Pred : Dataset
//    {
//        public List<string> cols;
//        public string data_path;
//        public Tensor data_stamp;
//        public Tensor data_x;
//        public Tensor data_y;
//        public string features;
//        public string freq;
//        public bool inverse;
//        public int label_len;
//        public int pred_len;
//        public string root_path;
//        public bool scale;
//        public int seq_len;
//        public string target;
//        public int timeenc;

//        public Dataset_Pred(
//            string root_path,
//            string flag = "pred",
//            int[] size = null,
//            string features = "S",
//            string data_path = "ETTh1.csv",
//            string target = "OT",
//            bool scale = true,
//            bool inverse = false,
//            int timeenc = 0,
//            string freq = "15min",
//            List<string> cols = null)
//        {
//            // size [seq_len, label_len, pred_len]
//            // info
//            if (size == null) {
//                this.seq_len = 24 * 4 * 4;
//                this.label_len = 24 * 4;
//                this.pred_len = 24 * 4;
//            } else {
//                this.seq_len = size[0];
//                this.label_len = size[1];
//                this.pred_len = size[2];
//            }
//            // init
//            Debug.Assert(new List<string> { "pred" }.Contains(flag));
//            this.features = features;
//            this.target = target;
//            this.scale = scale;
//            this.inverse = inverse;
//            this.timeenc = timeenc;
//            this.freq = freq;
//            this.cols = cols;
//            this.root_path = root_path;
//            this.data_path = data_path;
//            this.@__read_data__();
//        }

//        public virtual void @__read_data__()
//        {
//            List<string> cols;
        
//            var file = os.path.join(this.root_path, this.data_path);
//            using (var reader = new StreamReader(file))
//            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
//                var records = csv.GetRecords<ETT>();
//                if (this.cols != null) {
//                    cols = this.cols.copy();
//                    cols.remove(this.target);
//                } else {
//                    cols = csv.HeaderRecord.ToList();
//                    cols.remove(this.target);
//                    cols.remove("date");
//                }
//                var border1 = df_raw.Count - this.seq_len;
//                var border2 = df_raw.Count;


//                records = records.Skip(border1).Take(border2 - border1).ToList();
//                //if (this.scale) {
//                //    records = records.Skip(border1s[0]).Take(border2s[0]- border1s[0]).ToList();
//                //}  
//                List<Tensor> data = new List<Tensor>();
//                foreach (var item in records) {
//                    float[] datas = new float[7];
//                    datas[0] = (float)item.HUFL;
//                    datas[1] = (float)item.HULL;
//                    datas[2] = (float)item.MUFL;
//                    datas[3] = (float)item.MULL;
//                    datas[4] = (float)item.LUFL;
//                    datas[5] = (float)item.LULL;
//                    datas[6] = (float)item.OT;
//                    data.Add(datas);
//                }
//                var data2 = torch.cat(data);
//                this.data_x = data2;
//                this.data_y = data2;

//                this.size = data.Count;

//                var df_stamp = records.Select(q => q.date).ToList();
//                var df_stamp2 = Timefeatures.time_features(df_stamp, this.freq);
//                var df_stamp3 = new List<Tensor>();
//                foreach (var item in df_stamp2) { df_stamp3.Add(item); }
//                this.data_stamp = torch.cat(df_stamp3).transpose(1, 0);
//            }


//            object data;
//            object df_data;
//            this.scaler = new StandardScaler();
//            var df_raw = pd.read_csv(os.path.join(this.root_path, this.data_path));

      
//            df_raw = df_raw["date" + cols + this.target];
//            var border1 = df_raw.Count - this.seq_len;
//            var border2 = df_raw.Count;
//            if (this.features == "M" || this.features == "MS") {
//                var cols_data = df_raw.columns[1];
//                df_data = df_raw[cols_data];
//            } else if (this.features == "S") {
//                df_data = df_raw[new List<string> {
//                        this.target
//                    }];
//            }
//            if (this.scale) {
//                this.scaler.fit(df_data.values);
//                data = this.scaler.transform(df_data.values);
//            } else {
//                data = df_data.values;
//            }
//            var tmp_stamp = df_raw[new List<string> {
//                    "date"
//                }][TensorIndex.Slice(border1, border2, null)];
//            tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date);
//            var pred_dates = pd.date_range(tmp_stamp.date.values[^1], periods: this.pred_len + 1, freq: this.freq);
//            var df_stamp = pd.DataFrame(columns: new List<string> {                    "date"                });
//            df_stamp.date = tmp_stamp.date.values.ToList() + pred_dates[1].ToList();
//            if (this.timeenc == 0) {
//                df_stamp["month"] = df_stamp.date.apply(row => row.month, 1);
//                df_stamp["day"] = df_stamp.date.apply(row => row.day, 1);
//                df_stamp["weekday"] = df_stamp.date.apply(row => row.weekday(), 1);
//                df_stamp["hour"] = df_stamp.date.apply(row => row.hour, 1);
//                df_stamp["minute"] = df_stamp.date.apply(row => row.minute, 1);
//                df_stamp["minute"] = df_stamp.minute.map(x => x / 15);
//                var data_stamp = df_stamp.drop(new List<string> {
//                        "date"
//                    }, 1).values;
//            } else if (this.timeenc == 1) {
//                data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq: this.freq);
//                data_stamp = data_stamp.transpose(1, 0);
//            }
//            this.data_x = data[TensorIndex.Slice(border1, border2, null)];
//            if (this.inverse) {
//                this.data_y = df_data.values[TensorIndex.Slice(border1, border2, null)];
//            } else {
//                this.data_y = data[TensorIndex.Slice(border1, border2, null)];
//            }
//            this.data_stamp = data_stamp;
//        }


//        private int size;

//        public override long Count {
//            get
//            {
//                return this.size - this.seq_len - this.pred_len + 1;
//            }
//        }
//        public override Dictionary<string, Tensor> GetTensor(long index)
//        {
//            Tensor seq_y;
//            var s_begin = index;
//            var s_end = s_begin + this.seq_len;
//            var r_begin = s_end - this.label_len;
//            var r_end = r_begin + this.label_len + this.pred_len;
//            var seq_x = this.data_x[TensorIndex.Slice(s_begin, s_end, null)];
//            if (this.inverse) {
//                seq_y = this.data_x[TensorIndex.Slice(r_begin, (r_begin + this.label_len), null)];
//            } else {
//                seq_y = this.data_y[TensorIndex.Slice(r_begin, (r_begin + this.label_len), null)];
//            }
//            var seq_x_mark = this.data_stamp[TensorIndex.Slice(s_begin, s_end, null)];
//            var seq_y_mark = this.data_stamp[TensorIndex.Slice(r_begin, r_end, null)];
//            return new Dictionary<string, Tensor> {
//                { "batch_x",seq_x },
//                { "batch_y",seq_y },
//                { "batch_x_mark",seq_x_mark },
//                { "batch_y_mark",seq_y_mark },
//            };
//        }
//    }
//}
