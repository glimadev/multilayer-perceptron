using System;

namespace MLP_.Domain.TransferFunction
{
    public class HyperbolicTangentTransferFunction : ITransferFunction
    {
        public double Evaluate(double value)
        {
            // tanh - output range [-1.0..1.0]
            return Math.Tanh(value);
        }

        public double EvaluateDerivative(double value)
        {
            // tanh derivative
            return (1.0 - value * value);
        }
    }
}
