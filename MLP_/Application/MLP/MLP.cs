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
        double error;
        Random random = new Random();

        public MLP(int[] topology, ITransferFunction transferFunction)
        {
            _transferFunction = transferFunction;

            _net = new NeuralNet();

            CreateNetBasedOnTopology(topology);
        }

        private void CreateNetBasedOnTopology(int[] topology)
        {
            for (int i = 0; i < topology.Length; ++i)
            {
                int numNeurons = topology[i];

                bool isLastLayer = (i == (topology.Length - 1));

                // 0 output if on the last layer
                int numOutputs = ((isLastLayer) ? (0) : (topology[i + 1]));

                Layer layer = new Layer();

                // We have a new layer, now fill it with neurons, and
                // add a bias neuron in each layer.
                for (int j = 0; j < (numNeurons + 1); ++j)
                {
                    //Also Initialize random weights
                    layer.Neuron.Add(new Neuron(numOutputs, j, random));
                }

                // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
                layer.Neuron[numNeurons].Output = 1;

                _net.Layer.Add(layer);
            }
        }

        public void FeedForward(double[] input) //Layer prevLayer)
        {
            // Assign (latch) the input values into the input neurons
            for (int i = 0; i < input.Length; ++i)
            {
                _net.Layer[0].Neuron[i].Output = input[i];
            }

            // forward propagate
            for (int l = 1; l < _net.Layer.Count; ++l) // exclude input layer
            {
                Layer prevLayer = _net.Layer[l - 1];
                Layer currLayer = _net.Layer[l];

                int numNeuron = currLayer.Neuron.Count - 1; // exclude bias neuron

                for (int n = 0; n < numNeuron; ++n)
                {
                    currLayer.Neuron[n].FeedForward(prevLayer, _transferFunction);
                }
            }
        }

        public void BackPropagation(double[] targets)
        {
            // Calculate overall net error (RMS of output neuron errors)
            Layer outputLayer = _net.Layer.LastOrDefault();

            error = 0;

            for (int n = 0; n < (outputLayer.Neuron.Count - 1); ++n)
            {
                double delta = targets[n] - outputLayer.Neuron[n].Output;

                error += delta * delta;
            }

            error /= (outputLayer.Neuron.Count - 1);// get average error squared

            error = Math.Sqrt(error);//RMS

            // Implement a recent average measurement

            m_recentAvgError =
                    (m_recentAvgError * k_recentAvgSmoothingFactor + error)
                    / (k_recentAvgSmoothingFactor + 1.0);

            // Calculate output layer gradients
            for (int n = 0; n < (outputLayer.Neuron.Count - 1); ++n)
            {
                outputLayer.Neuron[n].CalcOutputGradients(targets[n], _transferFunction);
            }

            // Calculate hidden layer gradients
            for (int l = (_net.Layer.Count - 2); l > 0; --l)
            {
                Layer hiddenLayer = _net.Layer[l];
                Layer nextLayer = _net.Layer[l + 1];

                for (int n = 0; n < hiddenLayer.Neuron.Count; ++n)
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

        public double[] GetResults()
        {
            Layer lastLayer = _net.Layer.LastOrDefault();

            double[] resultVals = new double[lastLayer.Neuron.Count - 1];

            // exclude last neuron (bias neuron)
            for (int n = 0; n < (lastLayer.Neuron.Count - 1); ++n)
                resultVals[n] = lastLayer.Neuron[n].Output;

            return resultVals;
        }

        public void ShowVectorVals(string label, double[] v)
        {
            Console.Write(label + " ");

            for (int i = 0; i < v.Length; ++i)
                Console.Write(v[i] + " ");

            Console.WriteLine("\n");
        }

        public void Fit(List<double[]> trainData, List<double[]> targetData)
        {
            int trainingPass = 0;

            while (trainingPass < trainData.Count) 
            {
                Console.WriteLine("Pass " + trainingPass);

                // Get new input data and feed it forward:
                double[] inputVals = trainData.ElementAt(trainingPass);
                ShowVectorVals("Inputs:", inputVals);
                FeedForward(inputVals);

                // Collect the net's actual output results:
                double[] resultVals = GetResults();
                ShowVectorVals("Outputs:", resultVals);

                // Train the net what the outputs should have been:
                double[] targetVals = targetData.ElementAt(trainingPass);
                ShowVectorVals("Targets:", targetVals);
                BackPropagation(targetVals);

                // Report how well the training is working, average over recent samples:
                Console.WriteLine("Net current error: " + error);
                Console.WriteLine("Net recent average error: " + m_recentAvgError);

                if (trainingPass > 100 && m_recentAvgError < 0.05)
                {
                    Console.WriteLine("average error acceptable -> break " + m_recentAvgError);
                    break;
                }

                trainingPass++;
            }
        }
    }
}
