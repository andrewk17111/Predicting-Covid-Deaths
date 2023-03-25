using Experiment;
using NeuralNetwork;

const string OUTPUT = @"D:\";
const string INPUT = @"C:\Users\andre\OneDrive\Documents\School\SU\Winter Senior Research\AltInput\";
const string OUTPUT_TRAIN = OUTPUT + @"Train\";
const string OUTPUT_TEST = OUTPUT + @"Test\";

int[] LAYER_LENGTHS = { 10 };
const double BIAS = 1;
const double LEARNING_RATE = 0.05;
const int TREATMENTS = 30;
const int EPOCHS = 200;

double TESTING_SIZE = 0.10;
Func<double, double> ACTIVATION = Utils.Sigmoid;
Func<double, double> DERIVED_ACTIVATION = Utils.DerivedSigmoid;

foreach (string file in Directory.GetFiles(INPUT))
{
    if (file.Contains("state"))
    {
        Table<StateTerritory> table = new Table<StateTerritory>(file);

        RunTrial(table, file);
    }
    else if (file.Contains("division"))
    {
        Table<Division> table = new Table<Division>(file);

        RunTrial(table, file);
    }
    else if (file.Contains("region"))
    {
        Table<Region> table = new Table<Region>(file);

        RunTrial(table, file);
    }
    else
    {
        Table<Nation> table = new Table<Nation>(file);

        RunTrial(table, file);
    }
}

void RunTrial<T>(Table<T> table, string filename)
    where T : Enum
{
    Console.WriteLine(filename);

    File.Delete(OUTPUT_TRAIN + Path.GetFileName(filename));
    File.Delete(OUTPUT_TRAIN + "w" + Path.GetFileName(filename));
    File.Delete(OUTPUT_TEST + Path.GetFileName(filename));

    StreamWriter train_writer = new StreamWriter(OUTPUT_TRAIN + Path.GetFileName(filename));
    StreamWriter weights_writer = new StreamWriter(OUTPUT_TRAIN + "w" + Path.GetFileName(filename));
    StreamWriter test_writer = new StreamWriter(OUTPUT_TEST + Path.GetFileName(filename));

    train_writer.AutoFlush = true;
    weights_writer.AutoFlush = true;
    test_writer.AutoFlush = true;
    train_writer.WriteLine("Treatment,Epoch,Start,Stop,Time,Location,UntrainedOutput,TrainedOutput,Expected");
    weights_writer.WriteLine("Treatment,Epoch," + String.Join(",", Enumerable.Range(0, 34).Select((i) => $"w{i}")));
    test_writer.WriteLine("Treatment,Time,Location,Output,Expected");

    RunTreatments(table, train_writer, weights_writer, test_writer);

    train_writer.Close();
    weights_writer.Close();
    test_writer.Close();
}

void RunTreatments<T>(Table<T> table, StreamWriter train_writer, StreamWriter weights_writer, StreamWriter test_writer)
    where T : Enum
{
    for (int j = 0; j < TREATMENTS; j++)
    {
        NeuralNet net = new NeuralNet(LAYER_LENGTHS.Prepend(table.Width).ToArray(), BIAS, LEARNING_RATE, Utils.Sigmoid,
            Utils.DerivedSigmoid);

        table.ShuffleKeys();
        weights_writer.WriteLine($"{j},-1,-1,-1," + net.ToString());

        for (int i = 0; i < EPOCHS; i++)
        {
            Console.Write($"\r{i}");
            
            foreach ((DateTime, T) key in table.Keys[..^GetTestingSize(table.Keys.Length, TESTING_SIZE)])
            {
                double untrained_output = net.ForwardPass(table.Data[key]);
                long start = DateTime.Now.Ticks;
                double trained_output = net.Backpropagate(table.Data[key], table.Deaths[key]);
                long stop = DateTime.Now.Ticks;

                train_writer.WriteLine($"{j},{i},{start},{stop},{key.Item1},{key.Item2},{untrained_output},{trained_output}," +
                    $"{table.Deaths[key]}");
            }

            weights_writer.WriteLine($"{j},{i}," + net.ToString());
        }
        
        foreach ((DateTime, T) key in table.Keys[^GetTestingSize(table.Keys.Length, TESTING_SIZE)..])
            test_writer.WriteLine($"{j},{key.Item1.ToShortDateString()},{key.Item2}," +
                $"{net.ForwardPass(table.Data[key])},{table.Deaths[key]}");
    }
}

int GetTestingSize(int data_size, double testing_size)
    => Math.Max((int)(data_size * testing_size), 10);
