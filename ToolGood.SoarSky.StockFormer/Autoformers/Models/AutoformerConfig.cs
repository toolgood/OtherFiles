namespace ToolGood.SoarSky.StockFormer.Autoformers.Models
{
    public class AutoformerConfig
    {
        /// <summary>
        /// 移动平均窗口大小
        /// </summary>
        public int moving_avg = 25;

 

        /// <summary>
        ///时间特征编码的频率（默认为 h）。这可以设置为 s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly)。你也可以使用更详细的频率，如 15min 或 3h
        ///Freq for time features encoding (defaults to h). This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
        /// </summary>
        public virtual string freq { get; set; } = "d";
 
        #region forecasting task
        /// <summary>
        ///Informer 编码器的输入序列长度（默认为 96）
        ///Input sequence length of Informer encoder (defaults to 96)
        /// </summary>
        public int seq_len { get; set; } = 96;
        /// <summary>
        ///Informer 解码器的起始令牌长度（默认为 48）
        ///Start token length of Informer decoder (defaults to 48)
        /// </summary>
        public int label_len { get; set; } = 48;
        /// <summary>
        ///预测序列长度（默认为 24​​）
        ///Prediction sequence length (defaults to 24)
        /// </summary>
        public int pred_len { get; set; } = 96;

        #endregion

        #region model define

        /// <summary>
        ///编码器输入大小（默认为 7）
        ///Encoder input size (defaults to 7)
        /// </summary>
        public int enc_in { get; set; } = 7;
        /// <summary>
        ///解码器输入大小（默认为 7）
        ///Decoder input size (defaults to 7)
        /// </summary>
        public int dec_in { get; set; } = 7;
        /// <summary>
        ///输出大小（默认为 7）
        ///Output size (defaults to 7)
        /// </summary>
        public int c_out { get; set; } = 7;

        /// <summary>
        ///模型尺寸（默认为 512）
        ///Dimension of model (defaults to 512)
        /// </summary>
        public int d_model { get; set; } = 512;

        /// <summary>
        ///头数（默认为 8）
        ///Num of heads (defaults to 8)
        /// </summary>
        public int n_heads { get; set; } = 8;

        /// <summary>
        ///编码器层数（默认为 2）
        ///Num of encoder layers (defaults to 2)
        /// </summary>
        public int e_layers { get; set; } = 2;
        /// <summary>
        ///解码器层数（默认为 1）
        ///Num of decoder layers (defaults to 1)
        /// </summary>
        public int d_layers { get; set; } = 1;

        /// <summary>
        ///fcn 的维度（默认为 2048）
        ///Dimension of fcn (defaults to 2048)
        /// </summary>
        public int d_ff { get; set; } = 2048;

        /// <summary>
        ///Probsparse attn 因子（默认为 5）
        ///Probsparse attn factor (defaults to 5)
        /// </summary>
        public int factor { get; set; } = 1;


        /// <summary>
        ///辍学概率（默认为 0.05）
        ///The probability of dropout (defaults to 0.05)
        /// </summary>
        public double dropout { get; set; } = 0.05;
        /// <summary>
        ///时间特征编码（默认为 timeF）。这可以设置为timeF，fixed，learned, date
        ///Time features encoding (defaults to timeF). This can be set to timeF, fixed, learned
        /// </summary>
        public string embed { get; set; } = "timeF";

        /// <summary>
        ///是否在encoder中输出attention，使用该参数表示输出attention（默认为False）
        ///Whether to output attention in encoder, using this argument means outputing attention (defaults to False)
        /// </summary>
        public bool output_attention { get; set; } = false;
        /// <summary>
        ///激活函数（默认为gelu）
        ///Activation function (defaults to gelu)
        /// </summary>
        public string activation { get; set; } = "gelu";
        #endregion

        #region optimization

        /// <summary>
        ///Data loader 的 num_works（默认为 0）
        ///The num_works of Data loader (defaults to 0)
        /// </summary>
        public int num_workers { get; set; } = 10;
        /// <summary>
        ///训练时期（默认为 6）
        ///Train epochs (defaults to 6)
        /// </summary>
        public int train_epochs { get; set; } = 10;
        /// <summary>
        ///训练输入数据的批量大小（默认为 32）
        ///The batch size of training input data (defaults to 32)
        /// </summary>
        public int batch_size { get; set; } = 32;
        /// <summary>
        ///提前停止耐心（默认为 3）
        ///Early stopping patience (defaults to 3)
        /// </summary>
        public int patience { get; set; } = 3;
        /// <summary>
        ///优化器学习率（默认为 0.0001）
        ///Optimizer learning rate (defaults to 0.0001)
        /// </summary>
        public double learning_rate { get; set; } = 0.0001;
        /// <summary>
        ///调整学习率的方法（默认为 type1）
        ///Ways to adjust the learning rate (defaults to type1)
        /// </summary>
        public string lradj { get; set; } = "type1";

        #endregion
    }
}
