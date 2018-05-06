/*
 *    NaiveBayes.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
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

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.GaussianNumericAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NominalAttributeClassObserver;
import moa.classifiers.iss.ranking.InfoGainRanking;
import moa.classifiers.iss.ranking.MeanEuclideanDistanceRanking;
import moa.classifiers.iss.ranking.RankingFunction;
import moa.classifiers.iss.ranking.SymmetricUncertaintyRanking;
import moa.core.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Naive Bayes incremental learner with iterative subset selection.
 *
 * <p>
 *     Naive Bayes classifier with ISS feature selection
 * </p>
 *
 *
 * @author Lanqin Yuan
 * @version 1.0
 */
public class NaiveBayesISS extends AbstractClassifier
{
    private static final long serialVersionUID = 1L;

    public IntOption featureLimitOption = new IntOption( "featureLimit", 'f', "The number of top ranked features to conduct subset selection from, which is also the subset size upper bound. '-1' is a wildcard parameter which indicates all features in the stream.", -1, -1, Integer.MAX_VALUE);
    public IntOption reselectionIntervalOption = new IntOption( "reselectionInterval", 't', "The interval between when features are re-ranked.", 1000, 1, Integer.MAX_VALUE);
    public IntOption rankingWindowSizeOption = new IntOption( "rankingWindowSize", 'w', "The size of the window used for ranking.", 1000, 1, Integer.MAX_VALUE);
    public FloatOption decayFactorOption = new FloatOption("decayFactor", 'd', "The value of which to decay the accuracy estimates counts by at every decay interval.", 0.1, 0.0, 1.0);
    public IntOption decayIntervalOption = new IntOption("decayInterval", 'v', "The interval at which decay happens.", 1000, 1, Integer.MAX_VALUE);

    public MultiChoiceOption rankingOption = new MultiChoiceOption(
            "rankingMethod", 'm',
            "ranking function to use.",
            new String[]{"SU","InfoGain","MeanDistance"},
            new String[]{"Symmetric Uncertainty","Information gain","Average Euclidean distance"
            }, 0);


    // accuracy difference dump
    public StringOption outputNameOption = new StringOption("outputName",'n',"File name for output of accuracy difference as features are removed from subsets. An empty field produces no dump file.","");
    // buffered writer for writing out this dump file
    protected BufferedWriter bw;

    // window used for ranking of features
    protected Instances rankingWindow;
    protected RankingFunction rankingFunction = null;


    protected int rankedFeatureCount = 0;
    protected int[] bestFeatures;

    // correct and incorrect count for each class, used to rank subsets
    protected  int[] correctCount;
    protected  int[] wrongCount;
    protected double[] correctPercent;

    // 0 = subset of n features,
    // n-1 = subset of only the top feature
    protected int bestSubsetIndex = -1;

    // counters for reselection and decay which are incremented every instance
    protected int reselectionCounter = 0;
    protected int decayCounter = 0;

    // array to keep track of what each subset predicted
    protected int[] subsetClassPredictions;

    protected boolean initialised = false;

    protected DoubleVector observedClassDistribution;
    protected AutoExpandVector<AttributeClassObserver> attributeObservers;

    @Override
    public String getPurposeString() {
        return "Naive Bayes with ISS feature selection classifier.";
    }


    /**
     *
     * @param f
     */
    protected void initialiseFeatureSubsets(int f)
    {
        // might be bug if first instance is missing attributes TODO check
        // check if there is actually enough features
        // if there is less features overall than F (limit specified)
        if(f >= rankingWindow.numAttributes() || f == -1)
        {
            rankedFeatureCount = rankingWindow.numAttributes() - 1; // -1 as the class attribute should not be included
            if(rankedFeatureCount  < 0)
                rankedFeatureCount = 0; // really should never happen as there should always be at least 1 feature
        }
        else
        {
            rankedFeatureCount  = f;
        }


        // set first prediction as all features
        bestSubsetIndex = rankedFeatureCount - 1;

        // reset subset counts as subsets will be different
        correctCount = new int[rankingWindow.numAttributes()]; //int[featuresCount];
        wrongCount = new int[rankingWindow.numAttributes()]; //int[featuresCount];
        correctPercent = new double[rankingWindow.numAttributes()]; //double[featuresCount];

        // Only dump if filename is specified
        String fileName = outputNameOption.getValue();
        if (!fileName.equals(""))
        {
            try
            {
                File file = new File(fileName);

                // if file doesnt exists, then create it
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

    /**
     *  Select the best subset of features from active features.
     */
    protected void selectFeatureSubset()
    {

        // calculate accuracy gain from adding feature to subset
        // utilise accuracy difference by adding features to subset
        double[] accuracyArray = computeAccuracyDiff(correctPercent); // not currently used other than to dump

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
                bw.write(bestSubsetIndex + " out of " + rankingWindow.numAttributes());
                bw.write(System.lineSeparator());
                bw.flush();
            } catch (Exception e)
            {
                e.printStackTrace();
            }
        }

        // set best ranked features
        bestFeatures = rankingFunction.rankFeatures(rankingWindow, bestFeatures);
        //System.out.println(Arrays.toString(bestFeatures));
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
        bestFeatures = new int[rankedFeatureCount];
        for(int i = 0; i < bestFeatures.length;i++)
        {
            bestFeatures[i] = i;
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
        rankingFunction.initialise(rankedFeatureCount,rankingWindow.classIndex());
    }



    @Override
    public void resetLearningImpl()
    {
        this.rankingWindow = null;
        this.observedClassDistribution = new DoubleVector();
        this.attributeObservers = new AutoExpandVector<AttributeClassObserver>();
        this.initialised = false;
    }

    @Override
    public void setModelContext(InstancesHeader context)
    {
        try
        {
            this.rankingWindow = new Instances(context,0);
            this.rankingWindow.setClassIndex(context.classIndex());
        }
        catch(Exception e)
        {
            System.err.println("Error: no model Context available.");
            e.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst)
    {
        this.observedClassDistribution.addToValue((int) inst.classValue(), inst.weight());
        for (int i = 0; i < inst.numAttributes() - 1; i++)
        {
            int instAttIndex = modelAttIndexToInstanceAttIndex(i, inst);
            AttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs == null) {
                obs = inst.attribute(instAttIndex).isNominal() ? newNominalClassObserver()
                        : newNumericClassObserver();
                this.attributeObservers.set(i, obs);
            }
            obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
        }

        // if window is empty, initialise the window
        if (this.rankingWindow == null)
        {
            this.rankingWindow = new Instances(inst.dataset());
        }

        // ranking window and ranking function handling of new instance
        {

            // if window is full, delete last element in window
            if (this.rankingWindowSizeOption.getValue() <= this.rankingWindow.numInstances())
            {
                this.rankingWindow.delete(0);
                rankingFunction.removeInstance(this.rankingWindow.get(0));
            }
            // add element to window
            this.rankingWindow.add(inst);
            this.rankingFunction.addInstance(inst);
        }



        // feature selection
        // update best subset based on previous performance
        for(int i = 0; i < subsetClassPredictions.length;i++)
        {
            // if predicted value by this subset is equal to the true class value
            if(subsetClassPredictions[i] == (int)inst.classValue())
            {
                // increment correct count for that subset
                correctCount[i]++;
            }
            else
            {
                wrongCount[i]++;
            }

            // take care of class index
            if(i == inst.classIndex())
                correctPercent[i] = 0;
            else
                correctPercent[i] = (double)correctCount[i]/(double)(wrongCount[i] + correctCount[i]);
        }
        // update best subset based on new accuracy values
        bestSubsetIndex = Utils.maxIndex(correctPercent);
    }


    @Override
    public double[] getVotesForInstance(Instance inst)
    {
        // initialise stuff for first instance
        if (!initialised)
        {
            initialiseFeatureSubsets(featureLimitOption.getValue());
            initialiseRankingFunction();
            reselectionCounter = reselectionIntervalOption.getValue();
            decayCounter = decayIntervalOption.getValue();

            // add first instance to ranking function
            initialised = true;
        }
        else
        {
            // check if enough time as passed for re-ranking
            if(reselectionCounter <=0)
            {
                // get a new ranked list of features
                selectFeatureSubset();
                reselectionCounter = reselectionIntervalOption.getValue();
            }
            else
            {
                reselectionCounter--;
            }

            // check if enough time as passed decay
            if(decayCounter <=0)
            {
                // get a new ranked list of features
                decayCounts();
                decayCounter = decayIntervalOption.getValue();
            }
            else
            {
                decayCounter--;
            }
        }

        subsetClassPredictions = doNaiveBayesPrediction(inst, this.observedClassDistribution,this.attributeObservers,bestFeatures);
        double[] finalPrediction = new double[inst.numClasses()];
        finalPrediction[subsetClassPredictions[bestSubsetIndex]] = 1;

        //System.out.println(Arrays.toString(finalPrediction));
        // TODO check here
        return finalPrediction;
    }

    protected void decayCounts()
    {
        // decay counts based on option
        double decayFactor = decayFactorOption.getValue();
        for (int i = 0;i < rankedFeatureCount; i++)
        {
            correctCount[i] *= (1-decayFactor);
            wrongCount[i] *= (1-decayFactor);
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent)
    {
        for (int i = 0; i < this.observedClassDistribution.numValues(); i++) {
            StringUtils.appendIndented(out, indent, "Observations for ");
            out.append(getClassNameString());
            out.append(" = ");
            out.append(getClassLabelString(i));
            out.append(":");
            StringUtils.appendNewlineIndented(out, indent + 1,
                    "Total observed weight = ");
            out.append(this.observedClassDistribution.getValue(i));
            out.append(" / prob = ");
            out.append(this.observedClassDistribution.getValue(i)
                    / this.observedClassDistribution.sumOfValues());
            for (int j = 0; j < this.attributeObservers.size(); j++) {
                StringUtils.appendNewlineIndented(out, indent + 1,
                        "Observations for ");
                out.append(getAttributeNameString(j));
                out.append(": ");
                // TODO: implement observer output
                out.append(this.attributeObservers.get(j));
            }
            StringUtils.appendNewline(out);
        }
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    protected AttributeClassObserver newNominalClassObserver() {
        return new NominalAttributeClassObserver();
    }

    protected AttributeClassObserver newNumericClassObserver() {
        return new GaussianNumericAttributeClassObserver();
    }

    /**
     * returns an array of the class predicted at each subset size
     * @param inst instance to predict
     * @param observedClassDistribution
     * @param attributeObservers
     * @param rankedFeatureIndices indexes of ranked features in an array
     * @return
     */
    public static int[] doNaiveBayesPrediction(Instance inst,DoubleVector observedClassDistribution,AutoExpandVector<AttributeClassObserver> attributeObservers, int[] rankedFeatureIndices)
    {
        int[] classPrediction = new int[rankedFeatureIndices.length];
        double[] votes = getClassProb(observedClassDistribution);

        // do some stuff here to check if its ignored
        for (int i = 0; i < rankedFeatureIndices.length; i++)
        {
            for (int classIndex = 0; classIndex < votes.length; classIndex++)
            {
                int instAttIndex = modelAttIndexToInstanceAttIndex(rankedFeatureIndices[i],inst);
                AttributeClassObserver obs = attributeObservers.get(rankedFeatureIndices[i]);

                if ((obs != null) && !inst.isMissing(instAttIndex))
                {
                    votes[classIndex] *= obs.probabilityOfAttributeValueGivenClass(inst.value(instAttIndex), classIndex);
                }
            }
            // store prediction result
            classPrediction[i] = Utils.maxIndex(votes);
        }
        return classPrediction;
    }

    protected static double[] getClassProb(DoubleVector observedClassDistribution)
    {
        double[] classProb = new double[observedClassDistribution.numValues()];
        double observedClassSum = observedClassDistribution.sumOfValues();

        for (int classIndex = 0; classIndex < classProb.length; classIndex++)
        {
            classProb[classIndex] = observedClassDistribution.getValue(classIndex) / observedClassSum;
        }
        return classProb;
    }


    public void manageMemory(int currentByteSize, int maxByteSize)
    {
        // TODO Auto-generated method stub
    }
}
