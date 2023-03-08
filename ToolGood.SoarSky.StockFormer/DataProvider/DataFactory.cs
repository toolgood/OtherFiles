using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ToolGood.SoarSky.StockFormer.PatchTSTs.Exps;
using static TorchSharp.torch.utils.data;

namespace ToolGood.SoarSky.StockFormer.DataProvider
{
    public class DataFactory
    {
        public static Dictionary<string, Type> data_dict = new Dictionary<string, Type> {
            { "ETTh1", typeof( Dataset_ETT_hour)},
            { "ETTh2", typeof( Dataset_ETT_hour)},
            { "ETTm1", typeof( Dataset_ETT_minute)},
            { "ETTm2", typeof( Dataset_ETT_minute)},
            { "custom", typeof( Dataset_Custom)}};

        public static (Dataset, DataLoader) data_provider(IConfig args, string flag)
        {
            var Data = data_dict[args.data];
            var timeenc = args.embed != "timeF" ? 0 : 1;

            bool shuffle_flag, drop_last;
            int batch_size;
            string freq;
            Dataset data_set = null;

            if (flag == "test") {
                shuffle_flag = false;
                drop_last = true;
                batch_size = args.batch_size;
                freq = args.freq;
                //} else if (flag == "pred") {
                //    shuffle_flag = false;
                //    drop_last = false;
                //    batch_size = 1;
                //    freq = args.freq;
                //    Data = typeof(Dataset_Pred);
            } else {
                shuffle_flag = true;
                drop_last = true;
                batch_size = args.batch_size;
                freq = args.freq;
            }
            var size = new int[] { args.seq_len, args.label_len, args.pred_len };

            if (Data == typeof(Dataset_ETT_hour)) {
                data_set = new Dataset_ETT_hour(root_path: args.root_path, data_path: args.data_path, flag: flag, size: size,
                                                features: args.features, target: args.target, timeenc: timeenc, freq: freq);
            } else if (Data == typeof(Dataset_ETT_minute)) {
                data_set = new Dataset_ETT_minute(root_path: args.root_path, data_path: args.data_path, flag: flag, size: size,
                                                    features: args.features, target: args.target, timeenc: timeenc, freq: freq);
            } else if (Data == typeof(Dataset_Custom)) {
                data_set = new Dataset_Custom(root_path: args.root_path, data_path: args.data_path, flag: flag, size: size,
                                                features: args.features, target: args.target, timeenc: timeenc, freq: freq);
                //} else if (Data == typeof(Dataset_Pred)) {
                //    data_set = new Dataset_Pred(root_path: args.root_path, data_path: args.data_path, flag: flag, size: size,
                //                                features: args.features, target: args.target, timeenc: timeenc, freq: freq);
            }
            Console.WriteLine(flag, data_set.Count);
            var data_loader = new DataLoader(data_set, batchSize: batch_size, shuffle: shuffle_flag, num_worker: args.num_workers/*, drop_last: drop_last*/);
            return (data_set, data_loader);
        }

    }
}
