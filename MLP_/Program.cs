using MLP_.Application.Training;
using MLP_.Domain.TransferFunction;
using System;

namespace MLP_
{
    class Program
    {
        static void Main(string[] args)
        {
            TrainingData trainData = new TrainingData("Samples/out_xor.txt");

            MLP Net = new MLP(trainData.getTopology(), new HyperbolicTangentTransferFunction());

            Net.Fit(trainData.getTrainData(), trainData.getTargetData());

            double[] inputs = { 0, 1 };

            Console.ReadLine();
        }
    }
}
