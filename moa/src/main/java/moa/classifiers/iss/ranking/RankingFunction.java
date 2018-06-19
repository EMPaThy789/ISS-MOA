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

package moa.classifiers.iss.ranking;
import java.io.Serializable;
import java.io.StringReader;
import java.sql.Array;
import java.util.Arrays;
import java.util.List;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.lazy.neighboursearch.*;
import moa.core.Measurement;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import moa.core.Utils;

/**
 * Ranking function base class
 * @author Lanqin Yuan (fyempathy@gmail.com)
 * @version 1
 */
public abstract class RankingFunction implements Serializable
{
    // the threshold of accuracy difference needed before we take action(penalise/reward)
    protected double accuracyDifferenceConfidenceThresh = 1;
    // Squared value to use for comparison
    protected double accuracyDifferenceConfidenceThreshSquared = 1;
    // whether to only penalise features or to both penalise and reward
    protected boolean accuracyDifferencePenaliseOnly = true;
    // the weight to give the accuracy difference penalty/reward
    protected double accuracyDifferenceWeight = 0;

    protected int classIndex = -1; // uninitialised = -1
    protected int numberOfFeatures = -1;


    /**
     * initialises ranking function
     * @param numberOfFeatures
     * @param classIndex
     */
    public void initialise(int numberOfFeatures, int classIndex)
    {
        this.numberOfFeatures = numberOfFeatures;
        this.classIndex = classIndex;
    }

    /**
     * initialises ranking function
     * overload for when accuracy difference is enabled
     * @param numberOfFeatures
     * @param classIndex
     * @param accuracyDifferenceConfidenceThresh
     * @param accuracyDifferencePenaliseOnly
     * @param accuracyDifferenceWeight
     */
    public void initialise(int numberOfFeatures, int classIndex, double accuracyDifferenceConfidenceThresh,boolean accuracyDifferencePenaliseOnly, double accuracyDifferenceWeight)
    {
        this.accuracyDifferenceConfidenceThresh = accuracyDifferenceConfidenceThresh;
        this.accuracyDifferencePenaliseOnly = accuracyDifferencePenaliseOnly;
        this.accuracyDifferenceConfidenceThreshSquared = accuracyDifferenceConfidenceThresh * accuracyDifferenceConfidenceThresh;
        this.accuracyDifferenceWeight = accuracyDifferenceWeight;
        this.numberOfFeatures = numberOfFeatures;
        this.classIndex = classIndex;
    }


    /**
     * Ranks features
     * @param window
     * @return
     */
    public int[] rankFeatures(Instances window)
    {
        double[] rankingScoreArray = computeRankingScore(window);

        int[] rankedFeatures = sortFeatureArrayDesc(rankingScoreArray,numberOfFeatures);

        return rankedFeatures;
    }


    public int[] rankFeaturesAccuracyDifference(Instances window, double[] accuracyDifferenceArray)
    {
        double[] rankingScoreArray = computeRankingScore(window);

        // check that the accuracy difference actually is supplied (might not be on first instance)
        if(accuracyDifferenceArray != null)
        {
            double[] accuracyDiffScalar = new double[numberOfFeatures];
            // start at 1 to skip the subset containing a single feature
            for(int i = 1; i < accuracyDifferenceArray.length;++i)
            {

                // check for penalise only
                if(!accuracyDifferencePenaliseOnly || accuracyDifferenceArray[i] < 0)
                {
                    double accuracyDiffSq = accuracyDifferenceArray[i] * accuracyDifferenceArray[i];
                    // check if the difference is above the threshold
                    // use squared here to save the computation of sqrt
                    if(accuracyDiffSq > accuracyDifferenceConfidenceThreshSquared)
                    {
                        // scale the ranking score accordingly
                        rankingScoreArray[i] *= (1 + accuracyDifferenceArray[i] * accuracyDifferenceWeight);
                    }
                }
            }
        }

        int[] rankedFeatures = sortFeatureArrayDesc(rankingScoreArray,numberOfFeatures);
        return rankedFeatures;
    }

    /**
     * compute the ranking score for each feature using the given window.
     *
     * @param window
     * @return
     */
    protected abstract double[] computeRankingScore(Instances window);



    /**
     * adds an instance to the window
     * Implementation dependent on individual functions
     * @param inst instance to be added
     */
    public abstract void addInstance(Instance inst);

    /**
     * removes an instance from the window.
     * assumes that the instance has already been added before to the window as otherwise counts may go into negatives and crash stuff.
     * Implementation dependent on individual functions
     * @param inst instance to be removed
     */
    public abstract void removeInstance(Instance inst);



    /**
     * Selects the top 'numFeatures' number of features, sorted decendingly
     * @param scores An array containing the score of each feature. Its size should be the number of features
     * @param numFeatures the top number of features to return (the size of the returned array)
     * @return
     */
    public int[] sortFeatureArrayDesc(double[] scores, int numFeatures)
    {
        int[] returnArray = new int[numFeatures];
        for(int i = 0; i <  numFeatures; ++i)
        {
            returnArray[i] = -1;
        }

        for(int i = 0; i <  numFeatures; i++)
        {
            double largest= Double.NEGATIVE_INFINITY;
            int largestIndex = -1;
            for(int z = 0; z < scores.length;z++)
            {
                if(z!=classIndex)
                {
                    if (scores[z] > largest && !contains(returnArray, z))
                    {
                        largest = scores[z];
                        largestIndex = z;
                    }
                }
            }
            returnArray[i] = largestIndex;
        }

        return returnArray;
    }

    /**
     * checks if the index is in the array
     *
     * @param array
     * @param index
     * @return true if it is, false if no
     */
    private boolean contains (int[] array, int index)
    {
        for(int i = 0; i < array.length;i++)
        {
            if(array[i] == index)
                return true;
        }
        return false;
    }
}