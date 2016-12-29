using MLP_.Domain.TransferFunction;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_
{
    public class MLP : IMLP
    {
        NeuralNet _net;
        ITransferFunction _transferFunction;
        static double k_recentAvgSmoothingFactor = 100.00;
        double m_recentAvgError = 0;

        public MLP(int[] topology, ITransferFunction transferFunction) 
        {
            _transferFunction = transferFunction;

            _net = new NeuralNet();

            CreateNetBasedOnTopology(topology);
        }

        private void CreateNetBasedOnTopology(int[] topology)
        {
            for (int i = 0; i < topology.Length; i++)
            {
                int numNeurons = topology[i];

                bool isLastLayer = (i == (topology.Length - 1));

                // 0 output if on the last layer
                int numOutputs = ((isLastLayer) ? (0) : (topology[i + 1]));

                Layer layer = new Layer();

                // We have a new layer, now fill it with neurons, and
                // add a bias neuron in each layer.
                for (int j = 0; j < (numNeurons + 1); j++)
                {
                    //Also Initialize random weights
                    layer.Neuron.Add(new Neuron(numOutputs, j));
                }

                // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
                layer.Neuron[numNeurons].Output = 1;

                _net.Layer.Add(layer);
            }
        }

        public void FeedForward(double[] input) //Layer prevLayer)
        {
            // Assign (latch) the input values into the input neurons
            for (int i = 0; i < input.Length; i++)
            {
                _net.Layer[0].Neuron[i].Output = input[i];
            }
            
            // forward propagate
            for (int l = 1; l < _net.Layer.Count; l++) // exclude input layer
            {
                Layer prevLayer = _net.Layer[l - 1];
                Layer currLayer = _net.Layer[l];

                int numNeuron = currLayer.Neuron.Count - 1; // exclude bias neuron

                for (int n = 0; n < numNeuron; n++)
                {
                    currLayer.Neuron[n].FeedForward(prevLayer, _transferFunction);
                }
            }                
        }

        public void BackPropagation(double[] targets)
        {
            // Calculate overall net error (RMS of output neuron errors)
            Layer outputLayer = _net.Layer.LastOrDefault();

            double error = 0;

            for (int n = 0; n < (outputLayer.Neuron.Count - 1); n++)
            {
                double delta = targets[n] - outputLayer.Neuron[n].Output;
                
                error += delta * delta;
            }

            error = error / (outputLayer.Neuron.Count - 1);// get average error squared

            error = Math.Sqrt(error);//RMS

            // Implement a recent average measurement

            m_recentAvgError =
                    (m_recentAvgError * k_recentAvgSmoothingFactor + error)
                    / (k_recentAvgSmoothingFactor + 1.0);

            // Calculate output layer gradients
            for (int n = 0; n < (outputLayer.Neuron.Count - 1); n++)
            {
                outputLayer.Neuron[n].CalcOutputGradients(targets[n], _transferFunction);
            }

            // Calculate hidden layer gradients
            for (int l = (_net.Layer.Count - 2); l > 0; l--)
            {
                Layer hiddenLayer = _net.Layer[l];
                Layer nextLayer = _net.Layer[l + 1];

                for (int n = 0; n < hiddenLayer.Neuron.Count; n++)
                {
                    hiddenLayer.Neuron[n].CalcHiddenGradients(nextLayer, _transferFunction);
                }
            }

            // For all layers from outputs to first hidden layer,
            // update connection weights
            for (int i = (_net.Layer.Count - 1); i > 0; --i)
            {
                Layer currLayer = _net.Layer[i];
                Layer prevLayer = _net.Layer[i - 1];

                for (int n = 0; n < (currLayer.Neuron.Count - 1); ++n) // exclude bias
                    currLayer.Neuron[n].UpdateInputWeights(prevLayer);
            }
        }

        public void GetResults()
        {
            throw new NotImplementedException();
        }
    }
}
