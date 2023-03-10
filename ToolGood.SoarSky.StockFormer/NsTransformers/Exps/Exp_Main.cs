using ToolGood.SoarSky.StockFormer.DataProvider;
using ToolGood.SoarSky.StockFormer.NsTransformers.Models;
using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.utils.data;

namespace ToolGood.SoarSky.StockFormer.NsTransformers.Exps
{
    public class Exp_Main
    {
        public NsTransformer model;
        public Device device;
        public Exp_Config args;

        public Exp_Main(Exp_Config args)
        {
            device = cuda.is_available() ? CUDA : CPU;
            this.args = args;
            model = _build_model().to(device);
        }

        public virtual NsTransformer _build_model()
        {
            var model = new NsTransformer(args);
            return model;
        }

        public virtual (Dataset, DataLoader) _get_data(string flag)
        {
            var (data_set, data_loader) = DataFactory.data_provider(args, flag);
            return (data_set, data_loader);
        }

        public virtual OptimizerHelper _select_optimizer()
        {
            var model_optim = Adam(model.parameters(), lr: args.learning_rate);
            return model_optim;
        }
        public virtual Loss<Tensor, Tensor, Tensor> _select_criterion()
        {
            var criterion = nn.MSELoss();
            return criterion;
        }
        private (Tensor, Tensor, Tensor, Tensor) GetTensor(Dictionary<string, Tensor> dict)
        {
            var batch_x = dict["batch_x"];
            var batch_y = dict["batch_y"];
            var batch_x_mark = dict["batch_x_mark"];
            var batch_y_mark = dict["batch_y_mark"];
            return (batch_x, batch_y, batch_x_mark, batch_y_mark);
        }

        public virtual float vali(Dataset vali_data, DataLoader vali_loader, Loss<Tensor, Tensor, Tensor> criterion)
        {
            var total_loss = new List<float>();
            this.model.eval();
            using (var _no_grad = torch.no_grad()) {
                foreach (var dict in vali_loader) {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);

                    batch_x = batch_x.@float().to(this.device);
                    batch_y = batch_y.@float();
                    batch_x_mark = batch_x_mark.@float().to(this.device);
                    batch_y_mark = batch_y_mark.@float().to(this.device);
                    // decoder input
                    var dec_inp = torch.zeros_like(batch_y[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Colon]).@float();
                    dec_inp = torch.cat(new List<Tensor> { batch_y[TensorIndex.Colon, TensorIndex.Slice(null, this.args.label_len, null), TensorIndex.Colon], dec_inp }, dim: 1).@float().to(this.device);
                    // encoder - decoder
                    var outputs = this.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;
                    var f_dim = this.args.features == "MS" ? -1 : 0;
                    outputs = outputs[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Slice(f_dim, null)];
                    batch_y = batch_y[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Slice(f_dim, null)].to(this.device);
                    var pred = outputs.detach().cpu();
                    var @true = batch_y.detach().cpu();
                    var loss = criterion.forward(pred, @true);
                    total_loss.append(loss.item<float>());
                }
            }
            var total_loss2 = total_loss.Average();
            this.model.train();
            return total_loss2;
        }

        public virtual NsTransformer train(string setting)
        {
            Tensor outputs;
            var (train_data, train_loader) = this._get_data(flag: "train");
            var (vali_data, vali_loader) = this._get_data(flag: "val");
            var (test_data, test_loader) = this._get_data(flag: "test");
            var path = os.path.join(this.args.checkpoints, setting);
            if (!os.path.exists(path)) {
                os.makedirs(path);
            }
            var time_now = DateTime.Now;
            var train_steps = train_loader.Count;
            var early_stopping = new EarlyStopping(patience: this.args.patience, verbose: true);
            var model_optim = this._select_optimizer();
            var criterion = this._select_criterion();

            foreach (var epoch in Enumerable.Range(0, this.args.train_epochs)) {
                var iter_count = 0;
                var train_loss = new List<float>();
                this.model.train();
                var epoch_time = DateTime.Now;
                var i = -1;
                foreach (var dict in train_loader) {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                    i++;
                    iter_count += 1;
                    model_optim.zero_grad();
                    batch_x = batch_x.@float().to(this.device);
                    batch_y = batch_y.@float().to(this.device);
                    batch_x_mark = batch_x_mark.@float().to(this.device);
                    batch_y_mark = batch_y_mark.@float().to(this.device);
                    // decoder input
                    var dec_inp = torch.zeros_like(batch_y[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Colon]).@float();
                    dec_inp = torch.cat(new List<Tensor> { batch_y[TensorIndex.Colon, TensorIndex.Slice(null, this.args.label_len, null), TensorIndex.Colon], dec_inp }, dim: 1).@float().to(this.device);
                    // encoder - decoder
                    outputs = this.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;
                    var f_dim = this.args.features == "MS" ? -1 : 0;
                    outputs = outputs[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Slice(f_dim, null)];
                    batch_y = batch_y[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Slice(f_dim, null)].to(this.device);
                    var loss = criterion.forward(outputs, batch_y);
                    train_loss.append(loss.item<float>());

                    if ((i + 1) % 100 == 0) {
                        Console.WriteLine("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item<float>()));
                        var speed = (DateTime.Now - time_now) / iter_count;
                        var left_time = speed * ((this.args.train_epochs - epoch) * train_steps - i);
                        Console.WriteLine("\tspeed: {0:.4f}s/iter; left time: {1:.4f}s".format(speed, left_time));
                        iter_count = 0;
                        time_now = DateTime.Now;
                    }

                    loss.backward();
                    model_optim.step();
                }
                Console.WriteLine("Epoch: {0} cost time: {1}".format(epoch + 1, DateTime.Now - epoch_time));
                var train_loss2 = train_loss.Average();
                var vali_loss = this.vali(vali_data, vali_loader, criterion);
                var test_loss = this.vali(test_data, test_loader, criterion);
                Console.WriteLine("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss));
                early_stopping.__call__(vali_loss, this.model, path);
                if (early_stopping.early_stop) {
                    Console.WriteLine("Early stopping");
                    break;
                }
                adjust_learning_rate(model_optim, epoch + 1, this.args);
            }
            var best_model_path = path + "/" + "checkpoint.pth";
            this.model.save(best_model_path);
            //this.model.load_state_dict(torch.load(best_model_path));
            return this.model;
        }

        public virtual void test(string setting, bool test)
        {
            var (test_data, test_loader) = this._get_data(flag: "test");
            if (test) {
                Console.WriteLine("loading model");
                var path = os.path.join(this.args.checkpoints, setting);
                var best_model_path = path + "/" + "checkpoint.pth";
                this.model.load(best_model_path);
                //this.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")));
            }
            var preds = new List<Tensor>();
            var trues = new List<Tensor>();
            var folder_path = "./test_results/" + setting + "/";
            if (!os.path.exists(folder_path)) {
                os.makedirs(folder_path);
            }
            this.model.eval();
            using (var _no_grad = torch.no_grad()) {
                var i = -1;
                foreach (var dict in test_loader) {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                    i++;

                    batch_x = batch_x.@float().to(this.device);
                    batch_y = batch_y.@float().to(this.device);
                    batch_x_mark = batch_x_mark.@float().to(this.device);
                    batch_y_mark = batch_y_mark.@float().to(this.device);
                    // decoder input
                    var dec_inp = torch.zeros_like(batch_y[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Colon]).@float();
                    dec_inp = torch.cat(new List<Tensor> { batch_y[TensorIndex.Colon, TensorIndex.Slice(null, this.args.label_len, null), TensorIndex.Colon], dec_inp }, dim: 1).@float().to(this.device);
                    // encoder - decoder
                    var outputs = this.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;

                    var f_dim = this.args.features == "MS" ? -1 : 0;
                    outputs = outputs[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Slice(f_dim, null)];
                    batch_y = batch_y[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len, null), TensorIndex.Slice(f_dim, null)].to(this.device);
                    outputs = outputs.detach().cpu();//.numpy();
                    batch_y = batch_y.detach().cpu();//.numpy();
                    var pred = outputs;
                    var @true = batch_y;
                    preds.append(pred);
                    trues.append(@true);
                    if (i % 20 == 0) {
                        //input = batch_x.detach().cpu();//.numpy();
                        //gt = np.concatenate((input[0, TensorIndex.Colon, ^1], @true[0, TensorIndex.Colon, ^1]), axis: 0);
                        //pd = np.concatenate((input[0, TensorIndex.Colon, ^1], pred[0, TensorIndex.Colon, ^1]), axis: 0);
                        //visual(gt, pd, os.path.join(folder_path, i.ToString() + ".pdf"));
                    }
                }
            }
            var preds2 = torch.cat(preds);
            var trues2 = torch.cat(trues);
            Console.WriteLine("test shape:", preds2.shape, trues2.shape);
            preds2 = preds2.reshape(-1, preds2.shape[^2], preds2.shape[^1]);
            trues2 = trues2.reshape(-1, trues2.shape[^2], trues2.shape[^1]);
            Console.WriteLine("test shape:", preds2.shape, trues2.shape);
            // result save
            folder_path = "./results/" + setting + "/";
            if (!os.path.exists(folder_path)) {
                os.makedirs(folder_path);
            }

            var (mae, mse, rmse, mape, mspe, _, _) = Metrics.metric(preds2, trues2);
            Console.WriteLine("mse:{0}, mae:{1}".format(mse, mae));
            var fs = File.Open("result.txt", FileMode.OpenOrCreate);
            fs.Seek(0, SeekOrigin.End);
            var sw = new StreamWriter(fs);
            sw.Write(setting + "  \n");
            sw.Write("mse:{0}, mae:{1}".format(mse, mae));
            sw.WriteLine();
            sw.WriteLine();
            sw.Close();
            fs.Close();

            File.WriteAllText(folder_path + "metrics.npy",
                string.Join("\r\n", new List<float> {
                    mae.item<float>(),
                    mse.item<float>(),
                    rmse.item<float>(),
                    mape.item<float>(),
                    mspe.item<float>()
                }));
            File.WriteAllText(folder_path + "pred.npy", string.Join("\r\n", preds));
            File.WriteAllText(folder_path + "true.npy", string.Join("\r\n", trues));
        }

        public virtual void predict(string setting, bool load = false)
        {
            var (pred_data, pred_loader) = this._get_data(flag: "pred");
            if (load) {
                var path = os.path.join(this.args.checkpoints, setting);
                var best_model_path = path + "/" + "checkpoint.pth";
                this.model.load(best_model_path);
                //this.model.load_state_dict(torch.load(best_model_path));
            }
            var preds = new List<Tensor>();
            this.model.eval();
            using (var _no_grad = torch.no_grad()) {
                var i = -1;
                foreach (var dict in pred_loader) {
                    (var batch_x, var batch_y, var batch_x_mark, var batch_y_mark) = GetTensor(dict);
                    i++;

                    batch_x = batch_x.@float().to(this.device);
                    batch_y = batch_y.@float();
                    batch_x_mark = batch_x_mark.@float().to(this.device);
                    batch_y_mark = batch_y_mark.@float().to(this.device);
                    // decoder input
                    var dec_inp = torch.zeros(new long[] { batch_y.shape[0], this.args.pred_len, batch_y.shape[2] }).@float();
                    dec_inp = torch.cat(new List<Tensor> { batch_y[TensorIndex.Colon, TensorIndex.Slice(null, this.args.label_len, null), TensorIndex.Colon], dec_inp }, dim: 1).@float().to(this.device);
                    // encoder - decoder
                    var outputs = this.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;
                    var pred = outputs.detach().cpu();//.numpy();
                    preds.append(pred);
                }
            }
            var preds2=torch.cat(preds);
            preds2 = preds2.reshape(-1, preds2.shape[^2], preds2.shape[^1]);
            // result save
            var folder_path = "./results/" + setting + "/";
            if (!os.path.exists(folder_path)) {
                os.makedirs(folder_path);
            }
            File.WriteAllText(folder_path + "real_prediction.npy", string.Join("\r\n", preds));
            return;
        }

        public static void adjust_learning_rate(OptimizerHelper optimizer, int epoch, Exp_Config args)
        {
            Dictionary<int, double> lr_adjust = null;
            // lr = args.learning_rate * (0.2 ** (epoch // 2))
            if (args.lradj == "type1") {
                lr_adjust = new Dictionary<int, double> { { epoch, args.learning_rate * Math.Pow(0.5, (epoch - 1) / 1) } };
            } else if (args.lradj == "type2") {
                lr_adjust = new Dictionary<int, double> {
                    { 2, 5E-05},
                    { 4, 1E-05},
                    { 6, 5E-06},
                    { 8, 1E-06},
                    { 10, 5E-07},
                    { 15, 1E-07},
                    { 20, 5E-08}};
            }
            if (lr_adjust.keys().Contains(epoch)) {
                var lr = lr_adjust[epoch];
                foreach (var param_group in optimizer.ParamGroups) {
                    param_group.LearningRate = lr;
                    //param_group["lr"] = lr;
                }
                Console.WriteLine("Updating learning rate to {0}".format(lr));
            }
        }

    }
}
