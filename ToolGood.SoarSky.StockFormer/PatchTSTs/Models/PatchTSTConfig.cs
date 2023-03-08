namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Models
{
    public class PatchTSTConfig
    {
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


        #region PatchTST
        /// <summary>
        /// 完全连接的dropout
        /// fully connected dropout
        /// </summary>
        public double fc_dropout { get; set; } = 0.05;
        /// <summary>
        /// 头部 dropout
        /// </summary>
        public double head_dropout { get; set; } = 0.0;
        /// <summary>
        /// 补丁长度
        /// patch length
        /// </summary>
        public int patch_len { get; set; } = 16;
        public int stride { get; set; } = 8;
        /// <summary>
        /// None: None; end: padding on the end
        /// </summary>
        public string padding_patch { get; set; } = "end";
        /// <summary>
        /// RevIN
        /// </summary>
        public bool revin { get; set; } = true;
        /// <summary>
        /// RevIN-affine
        /// </summary>
        public bool affine { get; set; } = false;
        /// <summary>
        /// false：减去平均值；true：最后减去
        /// </summary>
        public bool subtract_last { get; set; } = false;
        /// <summary>
        /// 分解
        /// </summary>
        public bool decomposition { get; set; } = false;
        /// <summary>
        /// decomposition-kernel
        /// </summary>
        public int kernel_size { get; set; } = 25;
        /// <summary>
        /// individual head
        /// </summary>
        public bool individual { get; set; } = false;

        #endregion


        #region model define

        /// <summary>
        ///编码器输入大小（默认为 7）
        ///Encoder input size (defaults to 7)
        /// </summary>
        public int enc_in { get; set; } = 7;

        ///// <summary>
        /////输出大小（默认为 7）
        /////Output size (defaults to 7)
        ///// </summary>
        //public int c_out { get; set; } = 2;

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
        ///fcn 的维度（默认为 2048）
        ///Dimension of fcn (defaults to 2048)
        /// </summary>
        public int d_ff { get; set; } = 2048;


        /// <summary>
        ///辍学概率（默认为 0.05）
        ///The probability of dropout (defaults to 0.05)
        /// </summary>
        public double dropout { get; set; } = 0.05;

        #endregion




    }

}
