using System.Collections.Generic;

namespace MLP_
{
    public class NeuralNet
    {
        public NeuralNet()
        {
            Layer = new List<Layer>();
        }

        public List<Layer> Layer { get; set; }
    }
}
