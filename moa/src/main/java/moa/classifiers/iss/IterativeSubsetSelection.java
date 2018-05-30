package moa.classifiers.iss;

import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.iss.ranking.RankingFunction;

import java.io.BufferedWriter;

/**
 *
 */
public class IterativeSubsetSelection
{
    // accuracy difference
    public StringOption outputNameOption = new StringOption("outputName",'n',"File name for output of accuracy difference as features are removed from subsets. An empty field produces no dump file.","");
    // buffered writer for writing out this dump file
    protected BufferedWriter bw;

    // array of the current ranking. each array contains the index of the ranked feature. (e.g. [0] contains the index of the top ranked feature)
    protected int[] topRankedFeatureIndices;

    // bounds for hill climbing
    protected int lowerBound = 0;
    protected int upperBound = -1;


    // correct and incorrect count for each subset size, used to rank subsets
    protected  int[] correctCount;
    protected  int[] wrongCount;
    protected  double[] correctPercent;

    // array of prediction results for each subset size
    protected int[] subsetPredictionResult;

    // 0 = subset of n features, n-1 = subset of only the top feature
    protected int bestSubsetIndex = 0;
    protected int featuresCount = 0;

    // counters for reselection and decay
    protected int reselectionCounter = 0;
    protected int decayCounter = 0;
    protected int largestClassIndex = 0; // starts at 0 (0 = 1 class in the stream)

    protected boolean initialised = false;
    protected boolean hillClimbEnabled;
    protected int hillClimbWindow;

    // the ranking function
    protected RankingFunction rankingFunction = null;
    // the window of instances to use for ranking (this may be different to the window used by the classifier)
    protected Instances rankingWindow;
    protected int numberOfFeatures;

    public IterativeSubsetSelection(RankingFunction rankingFunction,boolean hillClimb, int hillClimbWindow,int numberOfFeatures)
    {
        this.rankingFunction = rankingFunction;
        this.hillClimbEnabled = hillClimb;
        this.hillClimbWindow = hillClimbWindow;
        this.numberOfFeatures = numberOfFeatures;
    }

    /**
     * gets the accuracy estimate for a subset
     * @param subsetIndex
     * @return
     */
    public double getAccuracyEstimate(int subsetIndex)
    {
        return correctPercent[subsetIndex];
    }




    /**
     * resets ISS for re-learning
     */
    public void resetLearningImpl()
    {
        initialised = false;
        topRankedFeatureIndices = null;
        rankingFunction = null;

        subsetPredictionResult = new int[numberOfFeatures];
        bestSubsetIndex = -1;

        lowerBound = 0;
        upperBound = -1;
        correctCount   = null;
        wrongCount     = null;
        correctPercent = null;
        featuresCount = 0;
        reselectionCounter = 0;
        decayCounter = 0;
        largestClassIndex = 0;
    }
}
