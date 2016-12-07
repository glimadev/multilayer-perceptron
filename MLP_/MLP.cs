using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP_
{
    public class MLP : IMLP
    {
        public NeuralNet Net { get; set; }
 
        public MLP(int[] topology)
        {
            Net = new NeuralNet();

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
                    layer.Neuron.Add(new Neuron(numOutputs));
                }

                // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
                layer.Neuron[numNeurons].Output = 1;

                Net.Layer.Add(layer);
            }
        }

        public void FeedForward(Layer prevLayer)
        {
            double sum = 0.0;

            // Sum the previous layer's outputs (which are our inputs)
            // Include the bias node from the previous layer.
            for (int k = 0; k < prevLayer.Neuron.Count; k++)
            {
                sum += prevLayer.Neuron[k].Output * prevLayer.Neuron[k].Connections[k].Weight;
            }
                
        }

        public void BackPropagation()
        {
            throw new NotImplementedException();
        }

        public void GetResults()
        {
            throw new NotImplementedException();
        }
    }
}
