using System;
using System.Collections.Generic;

namespace MLP_
{
    public class Neuron
    {
        public Neuron(int numOutputs)
        {
            Random random = new Random();

            Connections = new List<Connection>();

            for (int i = 0; i < numOutputs; i++)
            {
                Connections.Add(new Connection());
                Connections[i].Weight = random.NextDouble();
            }
        }

        public List<Connection> Connections { get; set; }

        public double Output { get; set; }
    }
}
