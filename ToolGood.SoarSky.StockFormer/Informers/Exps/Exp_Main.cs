using CsvHelper;
using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using ToolGood.SoarSky.StockFormer.DataProvider;
using ToolGood.SoarSky.StockFormer.Informers.Models;
using TorchSharp;
using static TorchSharp.torch.utils.data;
using TorchSharp.Modules;
using static TorchSharp.torch.optim.lr_scheduler;
using ToolGood.SoarSky.StockFormer.PatchTSTs.Exps;
using ToolGood.SoarSky.StockFormer.Utils;

namespace ToolGood.SoarSky.StockFormer.Informers.Exps
{
    public class Exp_Main
    {
        public Informer model;
        public Exp_Config args;
        public Device device;

        public Exp_Main(Exp_Config args)
        {
            device = cuda.is_available() ? CUDA : CPU;
            this.args = args;
            model = _build_model().to(device);
        }

        public virtual Informer _build_model()
        {
            return new Informer(this.args.enc_in, this.args.dec_in, this.args.c_out, this.args.seq_len, this.args.label_len, this.args.pred_len, this.args.factor, this.args.d_model,
                this.args.n_heads, this.args.e_layers, this.args.d_layers, this.args.d_ff, this.args.dropout, this.args.attn, this.args.embed, this.args.freq, this.args.activation,
                this.args.output_attention, this.args.distil, this.args.mix, this.device);//.@float();
        }

        public virtual (Dataset, DataLoader) _get_data(string flag)
        {
            var (data_set, data_loader) = DataFactory.data_provider(this.args, flag);
            return (data_set, data_loader);
        }

        public virtual OptimizerHelper _select_optimizer()
        {
            var model_optim = optim.Adam(this.model.parameters(), lr: this.args.learning_rate);
            return model_optim;
        }

        public virtual Loss<Tensor, Tensor, Tensor> _select_criterion()
        {
            var criterion = nn.MSELoss();
            return criterion;
        }

        public virtual float vali(Dataset vali_data, DataLoader vali_loader, Loss<Tensor, Tensor, Tensor> criterion)
        {
            this.model.eval();
            var total_loss = new List<float>();
            foreach (var dict in vali_loader) {
                var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                var (pred, @true) = this._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark);
                var loss = criterion.forward(pred.detach().cpu(), @true.detach().cpu());
                total_loss.append(loss.item<float>());
            }
            model.train();
            return total_loss.Average();
        }

        public virtual object train(string setting)
        {
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
                int i = -1;
                foreach (var dict in train_loader) {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                    i++;
                    iter_count += 1;
                    model_optim.zero_grad();
                    var (pred, @true) = this._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark);
                    var loss = criterion.forward(pred, @true);
                    train_loss.append(loss.cpu().item<float>());
                    if ((i + 1) % 100 == 0) {
                        Console.WriteLine("\titers: {0}, epoch: {1} | loss: {2}".format(i + 1, epoch + 1, loss.cpu().item<float>()));
                        var speed = (DateTime.Now - time_now).TotalSeconds / iter_count;
                        var left_time = speed * ((this.args.train_epochs - epoch) * train_steps - i);
                        Console.WriteLine("\tspeed: {0}s/iter; left time: {1}s".format(speed, left_time));
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
                Console.WriteLine("Epoch: {0}, Steps: {1} | Train Loss: {2} Vali Loss: {3} Test Loss: {4}".format(epoch + 1, train_steps, train_loss2, vali_loss, test_loss));
                early_stopping.__call__(vali_loss, this.model, path);
                if (early_stopping.early_stop) {
                    Console.WriteLine("Early stopping");
                    break;
                }
                adjust_learning_rate(model_optim, epoch + 1, this.args);
            }
            var best_model_path = path + "/" + "checkpoint.pth";
            //this.model.load_state_dict(torch.load(best_model_path));
            return this.model;
        }

        public virtual void test(string setting)
        {
            var (test_data, test_loader) = this._get_data(flag: "test");
            this.model.eval();
            var preds = new List<Tensor>();
            var trues = new List<Tensor>();
            foreach (var dict in test_loader) {
                var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                var (pred, @true) = this._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark);
                preds.append(pred.detach().cpu());
                trues.append(@true.detach().cpu());
            }

            //preds = np.array(preds);
            //trues = np.array(trues);
            //Console.WriteLine("test shape:", preds.shape, trues.shape);
            //preds = preds.reshape(-1, preds.shape[^2], preds.shape[^1]);
            //trues = trues.reshape(-1, trues.shape[^2], trues.shape[^1]);
            //Console.WriteLine("test shape:", preds.shape, trues.shape);
            //// result save
            //var folder_path = "./results/" + setting + "/";
            //if (!os.path.exists(folder_path)) {
            //    os.makedirs(folder_path);
            //}
            //(mae, mse, rmse, mape, mspe) = metric(preds, trues);
            //Console.WriteLine("mse:{}, mae:{}".format(mse, mae));
            //np.save(folder_path + "metrics.npy", np.array(new List<object> {
            //        mae,
            //        mse,
            //        rmse,
            //        mape,
            //        mspe
            //    }));
            //np.save(folder_path + "pred.npy", preds);
            //np.save(folder_path + "true.npy", trues);
        }

        public virtual void predict(string setting, bool load = false)
        {
            var (pred_data, pred_loader) = this._get_data(flag: "pred");
            if (load) {
                var path = os.path.join(this.args.checkpoints, setting);
                var best_model_path = path + "/" + "checkpoint.pth";
                this.model.load(best_model_path);
            }
            this.model.eval();
            var preds = new List<Tensor>();
            foreach (var dict in pred_loader) {
                var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                var (pred, @true) = this._process_one_batch(pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark);
                preds.append(pred.detach().cpu());
            }
            //var  preds2 = np.array(preds);
            //preds = preds.reshape(-1, preds.shape[^2], preds.shape[^1]);
            //// result save
            //var folder_path = "./results/" + setting + "/";
            //if (!os.path.exists(folder_path)) {
            //    os.makedirs(folder_path);
            //}
            //np.save(folder_path + "real_prediction.npy", preds);
            return;
        }

        public virtual (Tensor, Tensor) _process_one_batch(Dataset dataset_object, Tensor batch_x, Tensor batch_y,
                                                           Tensor batch_x_mark, Tensor batch_y_mark)
        {
            Tensor dec_inp = null;
            batch_x = batch_x.@float().to(this.device);
            batch_y = batch_y.@float();
            batch_x_mark = batch_x_mark.@float().to(this.device);
            batch_y_mark = batch_y_mark.@float().to(this.device);
            // decoder input
            if (this.args.padding == 0) {
                dec_inp = torch.zeros(new long[] { batch_y.shape[0], this.args.pred_len, batch_y.shape[^1] }).@float();
            } else if (this.args.padding == 1) {
                dec_inp = torch.ones(new long[] { batch_y.shape[0], this.args.pred_len, batch_y.shape[^1] }).@float();
            }
            dec_inp = torch.cat(new List<Tensor> { batch_y[TensorIndex.Colon, TensorIndex.Slice(null, this.args.label_len, null), TensorIndex.Colon], dec_inp }, dim: 1).@float().to(this.device);
            // encoder - decoder
            var outputs = this.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;

            //if (this.args.inverse) {
            //    outputs = dataset_object.inverse_transform(outputs);
            //}
            var f_dim = this.args.features == "MS" ? -1 : 0;
            batch_y = batch_y[TensorIndex.Colon, TensorIndex.Slice(-this.args.pred_len), TensorIndex.Slice(f_dim)].to(this.device);
            return (outputs, batch_y);
        }
        private (Tensor, Tensor, Tensor, Tensor) GetTensor(Dictionary<string, Tensor> dict)
        {
            var batch_x = dict["batch_x"];
            var batch_y = dict["batch_y"];
            var batch_x_mark = dict["batch_x_mark"];
            var batch_y_mark = dict["batch_y_mark"];
            return (batch_x, batch_y, batch_x_mark, batch_y_mark);
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
