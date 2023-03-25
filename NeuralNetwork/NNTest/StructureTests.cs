using Experiment;

namespace NNTest;

[TestClass]
public class StructureTests
{
    [TestMethod]
    public void OneToOne()
    {
        double[][] input = {
            new double[] { 0.0 },
            new double[] { 0.1 },
            new double[] { 0.2 },
            new double[] { 0.3 },
            new double[] { 0.4 },
            new double[] { 0.5 },
            new double[] { 0.6 },
            new double[] { 0.7 },
            new double[] { 0.8 },
            new double[] { 0.9 },
            new double[] { 1.0 }
        };
        double[] output = {
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        };
        NeuralNet net = new NeuralNet(new int[] { 1 }, 0.05, Utils.Linear, Utils.DerivedLinear);
        double diff = 1.0 - net.weights[0][0];
        
        for (int i = 0; i < 2000; i++)
            for (int j = 0; j < input.Length; j++)
                net.Backpropagate(input[j], output[j]);

        for (int i = 0; i < input.Length; i++)
            Assert.AreEqual(Math.Round(net.ForwardPass(input[i]), 10), output[i]);

        Assert.IsTrue(diff > 1.0 - net.weights[0][0]);
    }

    [TestMethod]
    public void OneToOneToOne()
    {
        double[][] input = {
            new double[] { 0.0 },
            new double[] { 0.1 },
            new double[] { 0.2 },
            new double[] { 0.3 },
            new double[] { 0.4 },
            new double[] { 0.5 },
            new double[] { 0.6 },
            new double[] { 0.7 },
            new double[] { 0.8 },
            new double[] { 0.9 },
            new double[] { 1.0 }
        };
        double[] output = {
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        };
        NeuralNet net = new NeuralNet(new int[] { 1, 1 }, 0.05, Utils.Linear, Utils.DerivedLinear);
        double[] diffs = net.weights.Select(x => 1.0 - x[0]).ToArray();

        for (int i = 0; i < 2000; i++)
            for (int j = 0; j < input.Length; j++)
            {
                net.ForwardPass(input[j]);
                net.Backpropagate(input[j], output[j]);
            }

        for (int i = 0; i < input.Length; i++)
            Assert.AreEqual(Math.Round(net.ForwardPass(input[i]), 10), output[i]);
            //Console.WriteLine(Math.Round(net.ForwardPass(input[i]), 10));

        Assert.IsTrue(net.weights.Select((x, i) => diffs[i] > (1.0 - x[0])).Aggregate((x, y) => x && y));
    }

    [TestMethod]
    public void Adding()
    {
        double[][] input = {
            new double[] { 0.363928756, 0.225266043 },
            new double[] { 0.148928982, 0.616809577 },
            new double[] { 0.312599978, 0.198347021 },
            new double[] { 0.558333528, 0.272137601 },
            new double[] { 0.688760206, 0.140483003 },
            new double[] { 0.353841507, 0.058211711 },
            new double[] { 0.356841533, 0.425527225 },
            new double[] { 0.395135394, 0.358309092 },
            new double[] { 0.008747945, 0.415031052 },
            new double[] { 0.340483057, 0.152661191 }
        };
        double[] output = {
            0.589194800,
            0.765738559,
            0.510946998,
            0.830471129,
            0.829243209,
            0.412053218,
            0.782368758,
            0.753444485,
            0.423778997,
            0.493144248
        };
        NeuralNet net = new NeuralNet(new int[] { 2 }, 0.05, Utils.Linear, Utils.DerivedLinear);

        for (int i = 0; i < 2000; i++)
            for (int j = 0; j < input.Length; j++)
            {
                net.ForwardPass(input[j]);
                net.Backpropagate(input[j], output[j]);
            }

        for (int i = 0; i < input.Length; i++)
            Assert.AreEqual(Math.Round(net.ForwardPass(input[i]), 5), Math.Round(output[i], 5));
    }

    [TestMethod]
    public void PredeterminedWeights()
    {
        double[][] input = {
            new double[] { 0.3969711485223440, 0.6241194227618960, 0.7025338773804310, 0.9304432865761750 },
            new double[] { 0.9152812098963750, 0.0589495491946477, 0.7771761250508660, 0.8323562288747890 },
            new double[] { 0.3858062828494790, 0.2324065502229610, 0.1254626091340760, 0.7201113609403280 },
            new double[] { 0.6331635595449740, 0.6299356018918460, 0.8029123617953190, 0.7859352373593270 },
            new double[] { 0.7125046770293020, 0.8886783617721790, 0.1101417935963230, 0.9010591806042660 },
            new double[] { 0.6848260032214610, 0.9160222851877110, 0.3571969123759760, 0.9251141055955410 },
            new double[] { 0.6489663659670690, 0.5108065592629130, 0.7672349442487450, 0.3543563812485250 },
            new double[] { 0.0056246522786514, 0.7881838145753310, 0.7790799554774110, 0.7234698300754860 },
            new double[] { 0.0716862689316253, 0.9298506032566720, 0.3205692054194690, 0.9843626365539520 },
            new double[] { 0.2788388683517790, 0.2469242687731940, 0.7969466863209500, 0.2432664930960050 },
            new double[] { 0.8667743807879230, 0.7924982188030800, 0.2915314466085270, 0.5427027681592450 },
            new double[] { 0.5207396239988640, 0.0192317578206309, 0.1093694546874570, 0.9998868295024500 },
            new double[] { 0.3399553440546070, 0.2721684702898050, 0.0746979288387262, 0.4215910299299110 },
            new double[] { 0.7032384432072950, 0.3494744677711770, 0.7161158402013690, 0.6383785829876370 },
            new double[] { 0.0190410983916456, 0.0138304479530433, 0.2465237084737570, 0.0150970882350223 },
            new double[] { 0.4124440923771930, 0.7411104925515480, 0.0032056144628851, 0.6291575037483490 },
            new double[] { 0.8366906836383630, 0.9999589964248800, 0.7161318890158680, 0.3658462982585490 },
            new double[] { 0.3038119086603040, 0.9471264994709640, 0.0942206812380577, 0.0147471539377446 },
            new double[] { 0.1020873092690810, 0.7700844846958300, 0.6536855123130410, 0.2705223991874130 },
            new double[] { 0.3162268486467980, 0.5772080849150740, 0.6208824606901320, 0.0710323843244551 },
            new double[] { 0.1228436487576290, 0.4926984512312740, 0.9806097639448030, 0.2772167060569200 },
            new double[] { 0.4205975339097570, 0.7324185874907920, 0.2539823179577310, 0.8585004460367100 },
            new double[] { 0.1477927061252950, 0.6178166680200330, 0.9633689252675870, 0.0312152527023322 },
            new double[] { 0.4210480033485610, 0.2396439788581090, 0.9639438652817330, 0.3819661295075640 },
            new double[] { 0.2377129284358550, 0.0307061948254563, 0.3297283794635780, 0.6783543548433380 },
            new double[] { 0.9126732936576070, 0.4382838069272550, 0.3451042453133700, 0.2592107419526690 },
            new double[] { 0.6067164484731980, 0.8911184184177490, 0.2627414504866460, 0.8858255605477690 },
            new double[] { 0.0880192108955433, 0.8861409275662380, 0.4742569063797110, 0.4434710878038850 },
            new double[] { 0.7480524295880740, 0.4562467738624790, 0.8292941155340270, 0.0477381001163846 },
            new double[] { 0.2739635666980200, 0.1410389594164880, 0.7227303721252270, 0.7843115210861690 },
            new double[] { 0.8838420193344660, 0.9316664545209990, 0.9937717922779850, 0.5475739580634480 },
            new double[] { 0.9156399309776580, 0.8733680296825710, 0.4590484318920800, 0.4355379064838630 },
            new double[] { 0.6886508607762950, 0.9648342201067340, 0.6353540621861370, 0.5129953481509950 },
            new double[] { 0.1000256165775240, 0.6986481510752160, 0.4806970150288020, 0.0606498110923626 },
            new double[] { 0.4805395186072560, 0.0952248161375397, 0.3613951143777970, 0.6444563996816630 },
            new double[] { 0.9444126106522680, 0.9727279455659760, 0.3281721024259300, 0.2197255691411630 },
            new double[] { 0.1857123827390220, 0.2912129552472680, 0.5227866324397250, 0.2586134703797720 },
            new double[] { 0.2330198632483100, 0.1773941633309240, 0.6569661863133460, 0.3825711457853000 },
            new double[] { 0.3861891714964540, 0.4357088841155900, 0.2508104580960420, 0.0888200840846527 },
            new double[] { 0.1693237685239760, 0.3947343610652940, 0.0293000906530141, 0.0533914496249037 }
        };
        double[] output =
        {
            0.5887236051666220,
            0.5883594529558770,
            0.5824308129188360,
            0.5897426706500140,
            0.5885078279214710,
            0.5899026003438630,
            0.5867813259199010,
            0.5868600101727170,
            0.5869130421339460,
            0.5829804085554790,
            0.5878904071163750,
            0.5834270399564340,
            0.5805089507506280,
            0.5874403326590600,
            0.5760611613374530,
            0.5841580788152910,
            0.5900840623220430,
            0.5818706597028370,
            0.5842139099137110,
            0.5830844352988750,
            0.5846242170145120,
            0.5866980212594290,
            0.5840207257766450,
            0.5853319199892230,
            0.5814199098036780,
            0.5850596297900820,
            0.5886839509929320,
            0.5847227694503230,
            0.5857275050804680,
            0.5848831234217970,
            0.5923009622533700,
            0.5888762206337230,
            0.5894854314573520,
            0.5817632695514430,
            0.5830643900239360,
            0.5877421058508480,
            0.5813215628228840,
            0.5823562916133800,
            0.5807971632475020,
            0.5779922237339280
        };
        NeuralNet net = new NeuralNet(new int[] { 4, 2 }, 0.05, Utils.Sigmoid, Utils.DerivedSigmoid);
        double[] diffs = Enumerable.Range(0, input.Length)
            .Select((i) => Math.Abs(output[i] - net.ForwardPass(input[i])))
            .ToArray();

        for (int i = 0; i < 4000; i++)
            for (int j = 0; j < input.Length; j++)
                net.Backpropagate(input[j], output[j]);

        Assert.IsTrue(
            input.Select((x, i) => diffs[i] > Math.Abs(output[i] - net.ForwardPass(x)) ? 1 : 0).Average() > 0.9);

        double[][] expected = {
            new double[] { 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2 },
            new double[] { 0.3, 0.3 }
        };

        net = new NeuralNet(new int[] { 4, 2 }, 0.05, Utils.Sigmoid, Utils.DerivedSigmoid, 
            expected);
        
        for (int i = 0; i < 4000; i++)
            for (int j = 0; j < input.Length; j++)
                net.Backpropagate(input[j], output[j]);

        Assert.IsTrue(net.weights[0]
            .Select((d, i) => Math.Round(d, 10) == Math.Round(expected[0][i], 10))
            .Aggregate((a, b) => a && b));
        Assert.IsTrue(net.weights[1]
            .Select((d, i) => Math.Round(d, 10) == Math.Round(expected[1][i], 10))
            .Aggregate((a, b) => a && b));
    }

    [TestMethod]
    public void AMNZ()
    {
        double[][] input = {
            new double[] { 0.1 },
            new double[] { 0.2 },
            new double[] { 0.3 },
            new double[] { 0.4 },
            new double[] { 0.5 },
            new double[] { 0.6 },
            new double[] { 0.7 },
            new double[] { 0.8 },
            new double[] { 0.9 },
            new double[] { 1 }
        };
        double[] output =
        {
            0.501305299,
            0.501019217,
            0.50079759,
            0.500626301,
            0.500494054,
            0.50039193,
            0.500312963,
            0.500251757,
            0.50020416,
            0.500166995
        };
        NeuralNet net = new NeuralNet(new int[] { 1, 1, 1, 1 }, 0.05, Utils.Sigmoid, Utils.DerivedSigmoid);
        double[] init_weights = net.weights.Select((w) => w[0]).ToArray();
        
        //for (int i = 0; i < 4000; i++)
            for (int j = 0; j < input.Length; j++)
                net.Backpropagate(input[j], output[j]);

        for (int i = 0; i < init_weights.Length; i++)
            Assert.AreNotEqual(net.weights[i][0], init_weights[i]);
    }
}
