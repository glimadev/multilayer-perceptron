using MLP_.Domain.TransferFunction;

namespace MLP_
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] topology = { 2, 3, 1 };

            double[] inputs = { 0, 1 };

            MLP Net = new MLP(topology, new HyperbolicTangentTransferFunction());

            Net.FeedForward(inputs);
        }
    }
}
