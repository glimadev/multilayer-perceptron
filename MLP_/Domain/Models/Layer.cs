using System.Collections.Generic;

namespace MLP_
{
    public class Layer
    {
        public Layer()
        {
            Neuron = new List<Neuron>();
        }

        public List<Neuron> Neuron { get; set; }
    }
}
