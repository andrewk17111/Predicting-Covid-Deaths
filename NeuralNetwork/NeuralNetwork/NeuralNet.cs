namespace NeuralNetwork;

public class NeuralNet
{
    private static readonly Random random = new Random();

    public readonly int InputLength;
    public readonly double Bias;
    public readonly double LearningRate;

    public Func<double, double> Activation
    {
        get;
        set;
    }
    public Func<double, double> DerivedActivation
    {
        get;
        set;
    }

    public double[][] weights;
    private double[] bias_weights;
    private double[][] nodes;

    public NeuralNet(int[] layers_lengths, double bias, double learning_rate,
        Func<double, double> activation, Func<double, double> derived_activation,
        double[][]? weights = null, double[]? bias_weights = null)
    {
        InputLength = layers_lengths.First();
        Bias = bias;
        LearningRate = learning_rate;
        Activation = activation;
        DerivedActivation = derived_activation;
        InitNodes(layers_lengths);

        if ((weights?.Length ?? 0) == 0)
            InitWeights();
        else
            this.weights = weights;

        if ((bias_weights?.Length ?? 0) == 0)
            InitBiasWeights();
        else
            this.bias_weights = bias_weights;
    }

    private void InitNodes(int[] layers_lengths)
    {
        nodes = new double[layers_lengths.Length + 1][];

        for (int i = 0; i < nodes.Length - 1; i++)
            nodes[i] = new double[layers_lengths[i]];

        nodes[^1] = new double[1];
    }

    private void InitWeights()
    {
        weights = new double[nodes.Length - 1][];

        for (int l = 0; l < weights.Length; l++)
        {
            weights[l] = new double[nodes[l].Length * nodes[l + 1].Length];

            for (int n = 0; n < weights[l].Length; n++)
                weights[l][n] = random.NextDouble();
        }
    }

    private void InitBiasWeights()
    {
        Random random = new Random();

        bias_weights = new double[weights.Length];

        for (int i = 0; i < bias_weights.Length; i++)
            bias_weights[i] = random.NextDouble();
    }

    public double ForwardPass(params double[] input)
    {
        if (input.Length == InputLength)
            nodes[0] = input;
        else
            throw new Exception();

        for (int l = 1; l < nodes.Length; l++)
            for (int n = 0; n < nodes[l].Length; n++)
                nodes[l][n] = CalculateNodeValue(l, n);

        return nodes.Last()[0];
    }

    public double Backpropagate(double[] input, params double[] expected)
    {
        double output = nodes.Last().Last();
        double[] cache = new double[1];
        double[][] new_weights = new double[weights.Length][];
        double[] new_bias_weights = new double[bias_weights.Length];

        // Train Output.
        new_weights[^1] = new double[weights.Last().Length];
        cache[0] = GetOutputError(nodes.Last()[0], expected[0]) * DerivedActivation(nodes.Last()[0]);

        for (int m = 0; m < nodes[^2].Length; m++)
        {
            int pos = GetWeightPos(nodes.Length - 2, m, 0);

            new_weights[^1][pos] = CalcNewWeightValue(weights.Last()[pos], cache[0] * nodes[^2][m]);
        }

        new_bias_weights[^1] = CalcNewWeightValue(bias_weights.Last(), cache[0] * Bias);

        // Train Hidden.
        for (int l = nodes.Length - 2; l > 0; l--)
        {
            double[] temp_cache = new double[nodes[l].Length];

            new_weights[l - 1] = new double[weights[l - 1].Length];

            for (int n = 0; n < nodes[l].Length; n++)
            {
                temp_cache[n] = cache.Select((d, i) => d * weights[l][GetWeightPos(l, n, i)]).Sum()
                    * DerivedActivation(nodes[l][n]);

                for (int m = 0; m < nodes[l - 1].Length; m++)
                {
                    int pos = GetWeightPos(l - 1, m, n);

                    new_weights[l - 1][pos] =
                        CalcNewWeightValue(weights[l - 1][pos], temp_cache[n] * nodes[l - 1][m]);
                }
            }

            cache = temp_cache;
            new_bias_weights[l] = CalcNewWeightValue(bias_weights[l], temp_cache.Sum() * Bias);
        }

        weights = new_weights;
        return ForwardPass(input);
    }

    private double GetOutputError(double output, double expected)
        => output - expected;

    private double CalcNewWeightValue(double weight_value, double error)
        => weight_value - LearningRate * error;

    private double CalculateNodeValue(int layer, int position)
    {
        double value = 0;

        for (int i = 0; i < nodes[layer - 1].Length; i++)
            value += nodes[layer - 1][i] * weights[layer - 1][GetWeightPos(layer - 1, i, position)];
        
        return Activation.Invoke(value + bias_weights[layer - 1] * Bias);
    }

    private int GetWeightPos(int start_layer, int start_position, int end_position)
        => nodes[start_layer].Length * end_position + start_position;

    public override string ToString()
        => String.Join(',', weights.Select((l) => String.Join(",", l)));
}
