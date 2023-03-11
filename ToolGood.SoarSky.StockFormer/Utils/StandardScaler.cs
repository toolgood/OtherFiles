using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.distributions.transforms;
using static TorchSharp.torch.utils;
using TorchSharp.Modules;

namespace ToolGood.SoarSky.StockFormer.Utils
{
    public class StandardScaler
    {
        private double mean;
        private double std;

        public StandardScaler() { }

        public StandardScaler(double mean, double std)
        {
            this.mean = mean;
            this.std = std;
        }
        // array([ 7.93774225,  2.02103866,  5.0797706 ,  0.74618588,  2.78176239,        0.78845312, 17.1282617 ])
        // array([33.78805569,  4.36853745, 30.45708257,  3.71093711,  1.04759863,        0.39719822, 84.20798753])

        public void fit(List<double> nums)
        {
            mean = nums.Average();
            var sum = 0.0;
            foreach (var num in nums) {
                sum += (num - mean) * (num - mean);
            }
            std = Math.Sqrt(sum / (nums.Count - 1));
        }
        public void fit(double[] nums)
        {
            mean = nums.Average();
            var sum = 0.0;
            foreach (var num in nums) {
                sum += (num - mean) * (num - mean);
            }
            std = Math.Sqrt(sum / (nums.Length - 1));
        }



        public double transform(double data)
        {
            return (data - this.mean) / this.std;
        }


        public double inverse_transform(double data)
        {
            return (data * this.std) + this.mean;
        }



    }


}
