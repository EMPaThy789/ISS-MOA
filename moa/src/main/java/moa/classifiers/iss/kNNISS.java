/*
 *    kNNBFE.java
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */

package moa.classifiers.iss;

import com.github.javacliparser.*;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.iss.knn.CumulativeLinearNNSearch;
import moa.classifiers.iss.ranking.InfoGainRanking;
import moa.classifiers.iss.ranking.MeanEuclideanDistanceRanking;
import moa.classifiers.iss.ranking.RankingFunction;
import moa.classifiers.iss.ranking.SymmetricUncertaintyRanking;
import moa.core.Measurement;
import moa.core.Utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * k Nearest Neighbor with Iterative Subset Selection.<p>
 *
 * Valid options are:<p>
 *
 * -k number of neighbours <br> -m max instances <br> -f number of best features to select subsets from <br> -t interval between when features are re-ranked
 *
 * @author Lanqin Yuan (fyempathy@gmail.com)
 * Orignial MOA kNN Jesse Read (jesse@tsc.uc3m.es)
 * @version 1.1
 */
public class kNNISS extends AbstractClassifier
{

    private static final long serialVersionUID = 2L; // some random number I entered (idk what this actually does tbh)

    // kNN parameters


    public IntOption kOption = new IntOption( "k", 'k', "The number of nearest neighbours.", 10, 1, Integer.MAX_VALUE);
    public IntOption windowSizeOption = new IntOption( "windowSize", 'w', "The maximum number of instances to store in the window.", 1000, 1, Integer.MAX_VALUE);

    // ISS parameters
    public IntOption featureLimitOption = new IntOption( "featureLimit", 'f', "The number of top ranked features to conduct subset selection from, which is also the subset size upper bound. '-1' is a wildcard parameter which indicates all features in the stream.", -1, -1, Integer.MAX_VALUE);
    public IntOption reselectionIntervalOption = new IntOption( "reselectionInterval", 't', "The interval between when features are re-ranked.", 1000, 1, Integer.MAX_VALUE);

    public IntOption decayIntervalOption = new IntOption("decayInterval", 'v', "The interval at which decay of accuracy estimate counts occur.", 1000, 1, Integer.MAX_VALUE);
    public FloatOption decayFactorOption = new FloatOption("decayFactor", 'd', "The value of which to decay accuracy estimate counts by.", 0.1, 0.0, 1.0);

    public FlagOption hillClimbOption = new FlagOption("hillCilmb", 'h', "Whether or not to select the feature limit via hill climbing.");
    public IntOption hillClimbWindowOption = new IntOption( "hillClimbWindow", 'a', "The size of the hill climb window.", 2, 0, 10);

    // public FloatOption accuracyGainWeightOption = new FloatOption("accuracyGainFactor", 'g', "How much weight to put into accuracy gain for ranking features.", 0.0, 0.0, 1.0);

    public MultiChoiceOption rankingOption = new MultiChoiceOption(
            "rankingMethod", 'm',
            "ranking function to use.",
            new String[]{"SU","InfoGain","MeanDistance"},
            new String[]{"Symmetric Uncertainty","Information gain","Average Euclidean distance"
        }, 0);

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
    protected int[] predictionOfSubset;

    // 0 = subset of n features, n-1 = subset of only the top feature
    protected int bestSubset = 0;
    protected int featuresCount = 0;

    // counters for reselection and decay
    protected int reselectionCounter = 0;
    protected int decayCounter = 0;
    protected int largestClassIndex = 0; // starts at 0 (0 = 1 class)

    protected boolean initialised = false;

    protected RankingFunction rankingFunction = null;
    protected Instances window;

    @Override
    public String getPurposeString() {
        return "kNNISS: kNN classifier with ISS feature selection.";
    }


	@Override
	public void setModelContext(InstancesHeader context)
    {
		try 
		{
			this.window = new Instances(context,0); //new StringReader(context.toString())
			this.window.setClassIndex(context.classIndex());
		}
		catch(Exception e)
        {
			System.err.println("Error: no model Context available.");
			e.printStackTrace();
			System.exit(1);
		}
	}

	/**
	 *  a method for initializing a classifier learner
	 */
    @Override
    public void resetLearningImpl()
    {
		initialised = false;
        this.window = null;
        topRankedFeatureIndices = null;
        rankingFunction = null;

        lowerBound = 0;
        upperBound = -1;
        correctCount   = null;
        wrongCount     = null;
        correctPercent = null;
        predictionOfSubset = null;
        bestSubset = 0;
        featuresCount = 0;
        reselectionCounter = 0;
        decayCounter = 0;
        largestClassIndex = 0;
    }

	/**
	 * a method to train a new instance
	 * @param inst new instance
     */
    @Override
    public void trainOnInstanceImpl(Instance inst) 
    {

        // sets C as the number of classes that a instance can be classified as and expands C if a new class value is seen
		if (inst.classValue() > largestClassIndex)
		{
            largestClassIndex = (int) inst.classValue();
        }

		// if window is empty, initialise the window
		if (this.window == null)
		{
			this.window = new Instances(inst.dataset());
		}

        // assign accuracy values for guess to subset
        // bounds should have been set in prediction
        for(int i = upperBound - 1; i >= lowerBound;i--)
        {
            // if predicted value by this subset is equal to the true class value
            if (predictionOfSubset[i] == (int)inst.classValue())
            {
                // increment correct count for that subset
                correctCount[i]++;
            }
            else
            {
                wrongCount[i]++;
            }
        }

        // calculate accuracy for selection of best subset
        for(int i = 0; i < correctPercent.length;i++)
        {
            correctPercent[i] = (double)correctCount[i]/(double)(wrongCount[i] + correctCount[i]);
        }
        // update best subset based on new accuracy values
        bestSubset = Utils.maxIndex(correctPercent);



		// updating sliding window
		// if window is full, delete last element in window
		if (this.windowSizeOption.getValue() <= this.window.numInstances())
		{
		    // also remove from ranking function
		    rankingFunction.removeInstance(this.window.get(0));
			this.window.delete(0);
		}

		// add element to window
		this.window.add(inst);
		// also add to ranking function
        rankingFunction.addInstance(inst);
    }

	/**
	 * a method to obtain the prediction result
	 * @param inst instance to predict the class value of
	 * @return array containing votes for the class values
     */
	@Override
    public double[] getVotesForInstance(Instance inst)
    {
        // vote array for class
		double v[] = new double[largestClassIndex+1];

        // check if enough time as passed
        if(reselectionCounter <=0)
        {
            // get a new ranked list of features
            selectFeatureSubset(featureLimitOption.getValue());
            reselectionCounter = reselectionIntervalOption.getValue();
        }
        else
        {
            reselectionCounter--;
        }

        // check if enough time as passed for decay
        if(decayCounter <=0)
        {
            // decay counts based on option
            double decayFactor = decayFactorOption.getValue();
            for (int i = 0;i < featuresCount; i++)
            {
                correctCount[i] *= (1-decayFactor);
                wrongCount[i] *= (1-decayFactor);
            }
            decayCounter = decayIntervalOption.getValue();
        }
        else
        {
            decayCounter--;
        }

		try
        {
            if(hillClimbOption.isSet())
            {
                // set upper and lower bound for hill climbing if it is set
                lowerBound = bestSubset - hillClimbWindowOption.getValue();
                if(lowerBound < 0)
                    lowerBound = 0;
                upperBound = bestSubset + hillClimbWindowOption.getValue() + 1;
                if(upperBound >= featuresCount)
                    upperBound = featuresCount;
            }
            else
            {
                // otherwise set
                lowerBound = 0;
                upperBound =  featuresCount;
            }

            // initialise search
            CumulativeLinearNNSearch cumulativeLinearNNSearch = new CumulativeLinearNNSearch();
            cumulativeLinearNNSearch.initialiseCumulativeSearch(inst, this.window, topRankedFeatureIndices,upperBound);

            // get a vote if there is enough instances in the window
            if (this.window.numInstances() > 0)
            {
                // backward elimination
                for(int z = upperBound - 1; z >= lowerBound;z--)
                {

                    // set number of features to consider in the search
                    cumulativeLinearNNSearch.setNumberOfActiveFeatures(z + 1);

                    // get knn search result
                    Instances neighbours = cumulativeLinearNNSearch.kNNSearch(inst, Math.min(kOption.getValue(), this.window.numInstances()));

                    // temp votes for current subset
                    double tempVotes[] = new double[largestClassIndex + 1];
                    for (int i = 0; i < neighbours.numInstances(); i++) {
                        tempVotes[(int) neighbours.instance(i).classValue()]++;
                    }

                    // set best subset as return for prediction before re-selecting best subset
                    if (bestSubset == z)
                    {
                        // save votes
                        v[Utils.maxIndex(tempVotes)] = 1;
                    }

                    // record prediction made by subset to use for learning via accuracy estimate
                    predictionOfSubset[z] = Utils.maxIndex(tempVotes);
                }
            }
		}
		catch(Exception e)
        {
			//System.err.println("Error: kNN search failed.");
			e.printStackTrace();
			//System.exit(1);
			return new double[inst.numClasses()];
		}
		return v;
    }

    /**
     *  Select the best subset of features from active features.
     * @param f Number of features specified
     */
    protected void selectFeatureSubset(int f)
    {
        // initialisation
        if (!initialised)
        {

            initialised = true;

            // might be bug if first instance is missing attributes TODO check
            // check if there is actually enough features
            // if there is less features overall than F (limit specified)
            // f = -1 is wildcard which selects all features
            if(f >= window.numAttributes() || f == -1)
            {
                featuresCount = window.numAttributes() - 1; // -1 as the class attribute should not be included
                if(featuresCount < 0)
                    featuresCount = 0; // really should never happen as there should always be at least 1 feature
            }
            else
            {
                featuresCount = f;
            }

            bestSubset = 0;
            // reset subset counts as subsets will be different
            correctCount = new int[window.numAttributes()];
            wrongCount = new int[window.numAttributes()];
            correctPercent = new double[window.numAttributes()];
            predictionOfSubset = new int[window.numAttributes()]; // we use one big array
            initialiseRankingFunction();
            // add first instance to ranking function


            // accuracy gain dump initialisation
            // Only dump if filename is specified
            String fileName = outputNameOption.getValue();
            if (!fileName.equals(""))
            {
                try
                {
                    File file = new File(fileName);

                    // if file doesn't exists, then create it
                    if (!file.exists())
                    {
                        file.createNewFile();
                    }

                    // write headers
                    bw = new BufferedWriter(new FileWriter(file.getAbsoluteFile()));
                    for (int i = 0; i < correctPercent.length; i++)
                    {
                        bw.write("Prediction Accuracy for subset of size" + (i + 1) + ",");
                    }
                    for (int i = 0; i < correctPercent.length; i++)
                    {
                        bw.write("Accuracy gain for subset of size " + (i + 1) + ",");
                    }
                    bw.write("Predicted number of relevant features out of total features,");

                    bw.write(featureLimitOption.getValue() + " Number of best ranked features considered");
                    bw.write(System.lineSeparator());

                }
                catch (IOException e)
                {
                    e.printStackTrace();
                }
            }
        }
        else // if already initialised
        {
            // calculate accuracy gain from adding feature to subset
            // utilise accuracy difference by adding features to subset
            writeAG(computeAccuracyDiff(correctPercent));
            // set best ranked features
            topRankedFeatureIndices = rankingFunction.rankFeatures(window, topRankedFeatureIndices);
        }
    }

    /**
     * writes to dump file for accuracy gain.
     */
    protected void writeAG(double[] accuracyArray)
    {
        // Write ag to file if specified
        if (bw != null)
        {
            try
            {
                for (int i = 0; i < correctPercent.length; i++)
                {
                    bw.write(Double.toString(correctPercent[i]) + ",");

                }
                for (int i = 0; i < accuracyArray.length; i++)
                {
                    bw.write(Double.toString(accuracyArray[i]) + ",");
                }
                bw.write(bestSubset + " out of " + window.numAttributes());
                bw.write(System.lineSeparator());
                bw.flush();
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
    }

    /**
     * Computes the prediction accuracy gained by adding the next best ranked feature F.
     * Done by comparing the difference in prediction accuracy of the subset containing F and the subset not containing F.
     * @param correctPercentage array of correct percentages with the best ranked feature being in index 0
     * @return array containing the prediction accuracy percentage gained or lost by adding the feature onto the subset
     */
    protected double[] computeAccuracyDiff(double[] correctPercentage)
    {
        double[] accuracyDiff = new double[correctPercentage.length];
        for (int i = 0; i < correctPercentage.length; i++)
        {
            if (i == 0)
                accuracyDiff[i] = 0;
            else {
                accuracyDiff[i] = correctPercentage[i] - correctPercentage[i - 1];
            }
        }
        return accuracyDiff;
    }

    /**
     * Initialises ranking function based on option set
     */
    public void initialiseRankingFunction()
    {
        // initialise best features as just the first k features
        topRankedFeatureIndices = new int[featuresCount];
        for(int i = 0; i < topRankedFeatureIndices.length;i++)
        {
            topRankedFeatureIndices[i] = i;
        }

        // initialise ranking function
        // select ranking function based on option
        switch (this.rankingOption.getChosenIndex())
        {
            case 0:
                rankingFunction = new SymmetricUncertaintyRanking();
                break;
            case 1:
                rankingFunction = new InfoGainRanking();
                break;
            case 2:
                rankingFunction = new MeanEuclideanDistanceRanking();
                break;
            default:
                break;
        }

        // initialise ranking function
        rankingFunction.initialise(featuresCount,window.classIndex());
    }


    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent)
    {

    }

    public boolean isRandomizable() {
        return false;
    }

}