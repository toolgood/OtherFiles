namespace ToolGood.SoarSky.StockFormer.DataProvider
{
    public interface IConfig
    {
        #region data loader

        string model { get; set; }// = "PatchTST";

        string data { get; set; }// = "ETTh1";
        /// <summary>
        ///数据文件的根路径（默认为 ./data/ETT/）
        ///The root path of the data file (defaults to ./data/ETT/)
        /// </summary>
        string root_path { get; set; }// = "./Datasets/ETT-small/";

        string data_path { get; set; }// = "ETTh1.csv";

        /// <summary>
        /// 预测任务，选项：[M，S，MS]；M： 多变量预测多变量，S：单变量预测单变量，MS：多变量预测单因素
        /// orecasting task, options:[M, S, MS]; M:multivariate predict multivariate,S:univariate predict univariate, MS:multivariate predict univariate
        /// </summary>
        string features { get; set; }// = "MS";

        string target { get; set; }// = "OT";

        /// <summary>
        ///时间特征编码的频率（默认为 h）。这可以设置为 s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly)。你也可以使用更详细的频率，如 15min 或 3h
        ///Freq for time features encoding (defaults to h). This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
        /// </summary>
        string freq { get; set; }// = "d";
        /// <summary>
        ///模型检查点的位置（默认为 ./checkpoints/）
        ///Location of model checkpoints (defaults to ./checkpoints/)
        /// </summary>
        string checkpoints { get; set; }// = "./checkpoints/";


        #endregion

        #region forecasting task
        /// <summary>
        ///Informer 编码器的输入序列长度（默认为 96）
        ///Input sequence length of Informer encoder (defaults to 96)
        /// </summary>
        int seq_len { get; set; }// = 96;
        /// <summary>
        ///Informer 解码器的起始令牌长度（默认为 48）
        ///Start token length of Informer decoder (defaults to 48)
        /// </summary>
        int label_len { get; set; }// = 48;
        /// <summary>
        ///预测序列长度（默认为 24​​）
        ///Prediction sequence length (defaults to 24)
        /// </summary>
        int pred_len { get; set; }// = 96;

        #endregion


        /// <summary>
        ///时间特征编码（默认为 timeF）。这可以设置为timeF，fixed，learned, date
        ///Time features encoding (defaults to timeF). This can be set to timeF, fixed, learned
        /// </summary>
        string embed { get; set; }// = "timeF";

        int batch_size { get; set; }

        int num_workers { get; set; }
    }
}
