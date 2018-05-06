package moa.classifiers.iss;

import com.github.javacliparser.StringOption;
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

    protected int[] topRankedFeatureIndices;

    // correct and incorrect count for each class, used to rank subsets
    protected  int[] correctCount;
    protected  int[] wrongCount;
    protected double[] correctPercent;

    // bounds for hill climbing
    protected int lowerBound = 0;
    protected int upperBound = -1;

    // array of prediction results for each subset size
    protected int[] subsetPredictionResult;

    // 0 = subset of n features, n-1 = subset of only the top feature
    protected int bestSubsetIndex = 0;
    protected int featuresCount = 0;

    // counters for reselection and decay
    protected int reselectionCounter = 0;
    protected int decayCounter = 0;
    protected int largestClassIndex = 0; // starts at 0 (0 = 1 class)

    protected boolean initialised = false;

    protected RankingFunction rankingFunction = null;

    public IterativeSubsetSelection(RankingFunction rankingFunction,boolean hillClimb, int hillClimbWindow)
    {

    }
}
