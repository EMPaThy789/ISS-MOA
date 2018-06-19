package moa.classifiers.iss.subsetselection;


/**
 * 4/06/2018
 * @author Lanqin Yuan
 */
public class ISSAccuracyEstimate
{
    private double decayMultiplier;

    private double[] correctCount;
    private double[] incorrectCount;

    // percent correct (correct/(correct+incorrect))
    private double[] accuracyEstimate;


    // total number of features in stream
    private int numberOfFeatures;
    private int classIndex;

    private boolean accuracyEstimateCurrent = false;


    public ISSAccuracyEstimate(int numberOfFeatures, double decayFactor,int classIndex)
    {
        // store only the multiplier
        decayMultiplier =  1 - decayFactor;
        this.numberOfFeatures = numberOfFeatures;
        correctCount = new double[numberOfFeatures];
        incorrectCount = new double[numberOfFeatures];
        this.classIndex = classIndex;
    }



    public double[] getAccuracyEstimates()
    {
        if(!accuracyEstimateCurrent)
        {
            accuracyEstimate = new double[numberOfFeatures];

            for(int i = 0; i < numberOfFeatures;++i)
            {
                // take care of class index
                if(i == classIndex)
                    accuracyEstimate[i] = 0;
                else
                    accuracyEstimate[i] = correctCount[i] / (correctCount[i] + incorrectCount[i]);

            }
            accuracyEstimateCurrent = true;
        }
        return accuracyEstimate;
    }

    /**
     * calculates the accuracy estimates and returns the index of the
     * subset size (actual subsets size -1) which has the highest estimate
     * @return
     */
    public int getBestSubsetSize()
    {
        int highestIndex = 0;
        double highestValue = Double.NEGATIVE_INFINITY;
        double[] accuracyEstimate = getAccuracyEstimates();
        for(int i = 0; i < numberOfFeatures;++i)
        {
            if(accuracyEstimate[i] > highestValue)
            {
                highestValue = accuracyEstimate[i];
                highestIndex = i;
            }
        }
        return highestIndex;
    }

    public void doDecay()
    {
        // decay all counts
        for(int i = 0; i < numberOfFeatures;++i)
        {
            correctCount[i] *= decayMultiplier;
            incorrectCount[i] *= decayMultiplier;
        }
        accuracyEstimateCurrent = false;
    }

    public void incrementCorrect(int subsetSizeIndex)
    {
        correctCount[subsetSizeIndex]++;
        accuracyEstimateCurrent = false;

    }

    public void incrementIncorrect(int subsetSizeIndex)
    {
        correctCount[subsetSizeIndex]++;
        accuracyEstimateCurrent = false;
    }


    /**
     * Computes the prediction accuracy gained by adding the next best ranked feature F.
     * Done by comparing the difference in prediction accuracy of the subset containing F and the subset not containing F.
     * AccuracyDifference(i) = AccuracyEstimate(i) - AccuracyEstimate(i-1)
     * @return array containing the prediction accuracy percentage gained or lost by adding the feature onto the subset
     */
    public double[] getAccuracyDiff()
    {
        double[] accuracyEstimate = getAccuracyEstimates();
        double[] accuracyDiff = new double[accuracyEstimate.length];
        for (int i = 0; i < accuracyEstimate.length; i++)
        {
            if (i == 0)
                accuracyDiff[i] = 0;
            else {
                accuracyDiff[i] = accuracyEstimate[i] - accuracyEstimate[i - 1];
            }
        }
        return accuracyDiff;
    }
}
