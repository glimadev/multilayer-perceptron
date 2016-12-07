using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] topology = { 2, 3, 1 };

            MLP Net = new MLP(topology);
        }
    }
}
