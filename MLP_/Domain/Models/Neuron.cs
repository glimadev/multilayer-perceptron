using MLP_.Domain.TransferFunction;
using System;
using System.Collections.Generic;

namespace MLP_
{
    public class Neuron
    {
        public List<Connection> Connections { get; set; }

        public double Output { get; set; }

        public double LocalGradient { get; set; }

        public int NeuronIndex { get; set; }

        public Neuron(int numOutputs, int j)
        {
            Random random = new Random();

            NeuronIndex = j;

            Connections = new List<Connection>();

            for (int i = 0; i < numOutputs; i++)
            {
                Connections.Add(new Connection());
                Connections[i].Weight = random.NextDouble();
            }
        }

        public void FeedForward(Layer prevLayer, ITransferFunction transferFunction)
        {
            double sum = 0.0;

            // Sum the previous layer's outputs (which are our inputs)
            // Include the bias node from the previous layer.
            for (int k = 0; k < prevLayer.Neuron.Count; k++)
            {
                sum += prevLayer.Neuron[k].Output * prevLayer.Neuron[k].Connections[NeuronIndex].Weight;
            }

            Output = transferFunction.Evaluate(sum);
        }

        public void CalcOutputGradients(double target, ITransferFunction transferFunction)
        {
            double delta = target - Output;

            LocalGradient = delta * transferFunction.EvaluateDerivative(delta);
        }

        internal void calcHiddenGradients(Layer nextLayer, ITransferFunction _transferFunction)
        {
            throw new NotImplementedException();
        }
    }
}
