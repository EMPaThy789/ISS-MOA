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
import com.yahoo.labs.samoa.instances.Instances;
import moa.core.Utils;

/**
 * SU ranking function
 * @author Lanqin Yuan (fyempathy@gmail.com)
 * @version 1
 */
public class SymmetricUncertaintyRanking extends InfoGainRanking
{
    private boolean debug = false;

    /**
     * Ranks features
     * @param window
     * @param previousBestFeatures
     * @return
     */
    @Override
    public int[] rankFeatures(Instances window, int[] previousBestFeatures)
    {
        double[] suArray = new double[window.numAttributes()];
        for(int a = 0;a < window.numAttributes(); a++)
        {
            if(a != window.classIndex())
                suArray[a] = computeSU(a,window);
            else
            {
                suArray[a] = Double.NEGATIVE_INFINITY;
                //System.out.println("class value su: " + computeSU(a, window));
            }
        }

        int[] rankedFeatures = sortFeatureArrayDesc(suArray,numberOfFeatures);

        /*System.out.println(Arrays.toString(suArray));
        System.out.println(Arrays.toString(rankedFeatures));*/
        return rankedFeatures;
    }


    /**
     * Compute the Symmetric uncertainty for the given attribute
     * @param a attribute index
     * @return
     */
    protected double computeSU(int a, Instances window)
    {
        double entropyBefore = 0;
        double entropyAfter = 0;
        double entropyAttribute = 0;

        // S = -k·sum[Pi log(Pi)]
        if(window.attribute(a).isNominal())
        {
            // nominal attribute
            NominalFeatureStats s = (NominalFeatureStats)attributeTableMap.get(a);
            // compute Entropy for all instances in the window
            for(int i = 0; i < window.attribute(a).numValues();i++)
            {
                for(int c = 0; c < window.numClasses();c++)
                {
                    if(s.classCount[c] > 0)
                    {
                        if(s.instCount > 0)
                        {
                            entropyAfter += (((double) s.varTotalCount[i] / (double) s.instCount)) * RankingUtils.computeEntropyNominal(s.varCount[i][c], s.varTotalCount[i]);
                            //System.out.println("total " + varTotalCount[i] + " inst count " + instCount + " prob " + Double.toString((double) varTotalCount[i] / (double) instCount) + " entropy " + Double.toString(RankingUtils.computeEntropyNominal(varCount[i][c], varTotalCount[i])));
                            //System.out.println("added " +  Double.toString((double) varTotalCount[i] / (double) instCount * RankingUtils.computeEntropyNominal(varCount[i][c], varTotalCount[i])));
                        }
                    }
                }
                // entropy attribute
                entropyAttribute += RankingUtils.computeEntropyNominal(s.varTotalCount[i],s.instCount);
            }
            //System.out.println("nominal entropy after for " + a + ": " + entropyAfter);

        }
        else
        if(window.attribute(a).isNumeric())
        {
            // numeric attribute processing

            NumericFeatureStats s = (NumericFeatureStats)attributeTableMap.get(a);

            // [bins][classes]
            int[][] counts = s.piD.generateContingencyTable();


            // compute for each bin
            for(int i = 0; i < counts.length;i++)
            {
                int total = Utils.sum(counts[i]);
                // compute entropy for each class
                for (int c = 0; c < counts[i].length; c++)
                {
                    if (counts[i][c] > 0)
                    {
                        if (s.instCount > 0)
                        {
                            entropyAfter += (((double) total / (double) s.instCount)) * RankingUtils.computeEntropyNominal(counts[i][c], total);
                            //System.out.println("var " + i + " class " + c +" total " + total + " inst count " + instCount + " prob " + Double.toString((double) total  / (double) instCount) + " entropy " + Double.toString(RankingUtils.computeEntropyNominal(counts[i][c], total)));
                            //System.out.println("added " +  Double.toString((double) total / (double) instCount * RankingUtils.computeEntropyNominal(counts[i][c], total)));

                        }
                    }
                }
                entropyAttribute += RankingUtils.computeEntropyNominal(Utils.sum(counts[i]),s.instCount);
            }
        }

        for(int i = 0; i < attributeTableMap.get(a).classCount.length; i++)
        {
            if(attributeTableMap.get(a).classCount[i] > 0)
            {
                double p = (double)attributeTableMap.get(a).classCount[i] / (double)attributeTableMap.get(a).instCount;
                entropyBefore += -p * Utils.log2(p);
            }
        }
        /*
        System.out.println("entropy before for " + a + ": " + entropyBefore);
        System.out.println("entropy attribute " + entropyAttribute);
        System.out.println("infogain " + (entropyBefore - entropyAfter));
        System.out.println("su " + 2*(entropyBefore - entropyAfter) / (entropyBefore + entropyAttribute));*/

        // return SU
        return 2*(entropyBefore - entropyAfter) / (entropyBefore + entropyAttribute);
    }

}