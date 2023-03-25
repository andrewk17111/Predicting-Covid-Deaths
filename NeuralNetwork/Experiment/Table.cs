namespace Experiment;

internal class Table<T>
    where T : Enum
{
    public readonly Dictionary<(DateTime, T), double[]> Data = new Dictionary<(DateTime, T), double[]>();
    public readonly Dictionary<(DateTime, T), double> Deaths = new Dictionary<(DateTime, T), double>();
    public (DateTime, T)[] Keys;
    public readonly int Width;
    
    private double max_location = Convert.ToDouble(((T[])Enum.GetValues(typeof(T))).Max());

    public Table(string path)
    {
        string[] rows = File.ReadAllLines(path);

        for (int i = 1; i < rows.Length; i++)
            ParseRow(rows[i]);

        Keys = Data.Keys.ToArray();
        Width = Data.First().Value.Length;
    }

    private void ParseRow(string row)
    {
        try
        {
            string[] cells = row.Split(',');
            DateTime date = DateTime.Parse(cells[0]);
            T location = (T)Enum.Parse(typeof(T), cells[2].Replace(" ", ""), true);

            // if (Convert.ToInt32(location) == 1) {
                // cells[2] = (Convert.ToDouble(location) / max_location).ToString();
                Data.Add((date, location), cells[3..].Select(Convert.ToDouble).ToArray());
                Deaths.Add((date, location), Convert.ToDouble(cells[1]));
            // }
        }
        catch (Exception ex)
        {
            Console.WriteLine(ex.Message);
            Console.WriteLine(ex.StackTrace);
            Environment.Exit(1);
        }
    }

    public void ShuffleKeys()
    {
        (DateTime, T)[] keys = Data.Keys.ToArray();
        Random random = new Random();

        for (int i = 0; i < keys.Length; i++)
        {
            int j = random.Next(keys.Length);
            
            (keys[j], keys[i]) = (keys[i], keys[j]);
        }

        Keys = keys;
    }
}
