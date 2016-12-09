
namespace MLP_.Domain.TransferFunction
{
    public interface ITransferFunction
    {
        double Evaluate(double value);
        double EvaluateDerivative(double value);
    }
}
