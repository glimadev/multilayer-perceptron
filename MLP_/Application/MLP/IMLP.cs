
namespace MLP_
{
    interface IMLP
    {
        void FeedForward(double[] inputs);
        void BackPropagation(double[] targets);
        double[] GetResults();
    }
}
