
namespace MLP_
{
    interface IMLP
    {
        void FeedForward(Layer prevLayer);
        void BackPropagation();
        void GetResults();
    }
}
