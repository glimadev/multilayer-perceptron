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

        static double eta = 0.15;   // [0.0..1.0] overall net training rate

        static double alpha = 0.5; // [0.0..n] multiplier of last weight change (momentum)

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

        public void CalcHiddenGradients(Layer nextLayer, ITransferFunction _transferFunction)
        {
            double dow = SumDOW(nextLayer);

            LocalGradient = dow * _transferFunction.EvaluateDerivative(Output);
        }

        private double SumDOW(Layer nextLayer)
        {
            double sum = 0.0;

            // Sum our contributions of the errors at the nodes we feed.
            int neurons = (nextLayer.Neuron.Count -1); // exclude bias neuron

            for (int n = 0; n < neurons; ++n)
                sum += Connections[n].Weight * nextLayer.Neuron[n].LocalGradient;

            return sum;
        }

        public void UpdateInputWeights(Layer prevLayer)
        {
            // The weights to be updated are in the Connection container
            // in the neurons in the preceding layer

            for (int n = 0; n < prevLayer.Neuron.Count; ++n)
            {
                Neuron neuron = prevLayer.Neuron[n];

                double oldDeltaWeight = neuron.Connections[NeuronIndex].DeltaWeight;

                double newDeltaWeight =
                    // Individual input, magnified by the gradient and train rate:
                        eta
                        * neuron.Output
                        * LocalGradient
                    // Also add momentum = a fraction of the previous delta weight;
                        + alpha
                        * oldDeltaWeight
                        ;

                neuron.Connections[NeuronIndex].DeltaWeight = newDeltaWeight;
                neuron.Connections[NeuronIndex].Weight += newDeltaWeight;
            }
        }
    }
}
