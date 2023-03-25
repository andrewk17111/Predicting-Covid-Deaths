namespace Experiment;

public static class Utils
{
    static readonly decimal E = Convert.ToDecimal(Math.E);

    /// <summary>
    /// Returns e raised to the specified power.
    /// Based on https://github.com/nathanpjones/DecimalMath/blob/master/DecimalEx/DecimalEx.cs.
    /// </summary>
    /// <param name="d">A number specifying a power.</param>
    //public static decimal Exp(decimal d)
    //{
    //    d = Math.Abs(d);
        
    //    decimal result;
    //    decimal nextAdd;
    //    int iteration;
    //    bool is_neg = d < 0;
    //    int int_pow = (int)Decimal.Truncate(d);
        
    //    if (d == 0)
    //    {
    //        result = 1;
    //    }
    //    else if (d == 1)
    //    {
    //        result = E;
    //    }
    //    else if (d == int_pow)
    //    {
    //        result = ExpBySquaring(E, d);
    //    }
    //    else if (Math.Abs(d) > 1)
    //    {
    //        // Split up into integer and fractional
    //        result = Exp(int_pow) * Exp(d - int_pow);
    //    }
    //    else
    //    {
    //        // See http://mathworld.wolfram.com/ExponentialFunction.html
    //        iteration = 0;
    //        nextAdd = 0;
    //        result = 0;

    //        while (true)
    //        {
    //            if (iteration == 0)
    //            {
    //                // == Pow(d, 0) / Factorial(0) == 1 / 1 == 1
    //                nextAdd = 1;
    //            }
    //            else
    //            {
    //                // == Pow(d, iteration) / Factorial(iteration)
    //                nextAdd *= d / iteration;
    //            }

    //            if (nextAdd == 0)
    //                break;

    //            result += nextAdd;
    //            iteration += 1;
    //        }
    //    }

    //    // Deals with negative powers.
    //    return is_neg
    //        ? (1 / result)
    //        : result;
    //}

    /// <summary>
    /// Raises one number to an integral power.
    /// Based on https://github.com/nathanpjones/DecimalMath/blob/master/DecimalEx/DecimalEx.cs.
    /// </summary>
    /// <remarks>
    /// See http://en.wikipedia.org/wiki/Exponentiation_by_squaring
    /// </remarks>
    //private static decimal ExpBySquaring(decimal x, decimal y)
    //{
    //    if (y < 0)
    //        throw new ArgumentOutOfRangeException("y", "Negative exponents not supported!");
        
    //    if (Decimal.Truncate(y) != y)
    //        throw new ArgumentException("Exponent must be an integer!", "y");

    //    decimal result = 1M;
    //    decimal multiplier = x;

    //    while (y > 0)
    //    {
    //        if (y % 2 == 1)
    //        {
    //            result *= multiplier;
    //            y -= 1;
                
    //            if (y == 0)
    //                break;
    //        }

    //        multiplier *= multiplier;
    //        y /= 2;
    //    }

    //    return result;
    //}

    //public static decimal Linear(decimal x)
    //    => x;

    //public static decimal DerivedLinear(decimal x)
    //    => 1;

    //public static decimal Sigmoid(decimal x)
    //    => 1 / (1 + Utils.Exp(-x));

    //public static decimal DerivedSigmoid(decimal x)
    //    => x * (1 - x);

    public static double Linear(double x)
        => x;

    public static double DerivedLinear(double x)
        => 1;

    const double SCALAR = 1;

    public static double Sigmoid(double x)
        => SCALAR / (1 + Math.Exp(-x / SCALAR));

    public static double DerivedSigmoid(double x)
        => x * (SCALAR - x);
}
