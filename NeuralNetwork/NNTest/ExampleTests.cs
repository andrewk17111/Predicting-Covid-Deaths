using Experiment;

namespace NNTest;

[TestClass]
public class ExampleTests
{
    /// <summary>
    /// A neural network example from TowardDataScience where the weights used to generate the dataset are provided.
    /// https://towardsdatascience.com/how-to-train-a-neural-network-from-scratch-952bbcdae729
    /// </summary>
    [TestMethod]
    public void TowardDataScience()
    {
        double[][] input = {
            new double[] { 3, 4, 7 },
            new double[] { 6, 2, 9 },
            new double[] { 8, 10, 12 },
            new double[] { 9, 7, 4 },
            new double[] { 7, 6, 3 }
        };
        double[] output = { 13.5, 15.6, 27.4, 15.6, 12.5 };
        NeuralNet net = new NeuralNet(new int[] { 3 }, 0.001, (x) => x, (x) => 1,
            new double[] { 20, -14, 346 });
        double[] target = { 0.5, 0.9, 1.2 };
        double[] diff = target.Select((d, i) => Math.Abs(d - net.weights[0][i])).ToArray();
        
        for (int j = 0; j < 2000; j++)
            for (int i = 0; i < input.Length; i++)
            {
                net.ForwardPass(input[i]);
                net.Backpropagate(input[i], output[i]);
            }

        double[] new_diff = target.Select((d, i) => Math.Abs(d - net.weights[0][i])).ToArray();

        Assert.IsTrue(diff.Select((d, i) => d > new_diff[i]).Aggregate((a, b) => a && b));
        Assert.AreEqual(Math.Round(net.ForwardPass(12, 10, 9), 1), 25.8);
    }

    /// <summary>
    /// The same neural network from TowardDataScience, but the network is preset to the predetermined weights allowing
    /// to test if the network still trains when the optimal weights are already found.
    /// </summary>
    [TestMethod]
    public void TowardDataSciencePredetermined()
    {
        double[][] input = {
            new double[] { 3, 4, 7 },
            new double[] { 6, 2, 9 },
            new double[] { 8, 10, 12 },
            new double[] { 9, 7, 4 },
            new double[] { 7, 6, 3 }
        };
        double[] output = { 13.5, 15.6, 27.4, 15.6, 12.5 };
        NeuralNet net = new NeuralNet(new int[] { 3 }, 0.001, (x) => x, (x) => 1,
            new double[] { 0.5, 0.9, 1.2 });
        double[] target = { 0.5, 0.9, 1.2 };
        
        for (int j = 0; j < 2000; j++)
            for (int i = 0; i < input.Length; i++)
            {
                net.ForwardPass(input[i]);
                net.Backpropagate(input[i], output[i]);
            }

        Assert.AreEqual(net.weights[0].Select((w, i) => Math.Abs(w - target[i])).Aggregate((a, b) => a + b), 0);
        Assert.AreEqual(Math.Round(net.ForwardPass(12, 10, 9), 1), 25.8);
    }

    /// <summary>
    /// A neural network example from Matt Mazur's backpropagation example, though in a modified form due to my network
    /// not being able to produce more than one output and not using biases.
    /// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    /// </summary>
    [TestMethod]
    public void MattMazur()
    {
        NeuralNet net = new NeuralNet(new int[] { 2, 2 }, 0.5, Utils.Sigmoid, Utils.DerivedSigmoid,
            new double[] { 0.15, 0.2, 0.25, 0.30 }, new double[] { 0.40, 0.45 });
        double untrained = net.ForwardPass(5, 10);
        const double EXPECTED = 1;

        for (int j = 0; j < 100000; j++)
            net.Backpropagate(new double[] { 5, 10 }, EXPECTED);

        Assert.IsTrue(Math.Abs(EXPECTED - net.ForwardPass(5, 10)) < Math.Abs(EXPECTED - untrained));
    }
}
