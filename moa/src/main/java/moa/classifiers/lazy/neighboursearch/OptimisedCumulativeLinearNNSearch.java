/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    CumulativeLinearNNSearch.java
 *    Copyright (C) 1999-2012 University of Waikato
 */

package moa.classifiers.lazy.neighboursearch;


import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.core.Utils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Class implementing cumulative sums of features to the brute force search algorithm for nearest neighbour search as an optimisation for feature selection of feature subsets.
 * @TODO This class should eventually replace the cumulative linear search class once it has been made sure that it gives the same results
 * @author Lanqin Yuan (fyempathy@gmail.com)
 * @version 01.2016
 */
public class OptimisedCumulativeLinearNNSearch extends CumulativeLinearNNSearch
{

    private double[] maxDist;

    private List<Inst> distanceUpdateList;
    private List<Inst> searchList;
    private Instance target;

    private int itterationsLeft = 0;

    private int numberOfActiveFeatures = -1;
    private double previousPivotDist = -1;



    public OptimisedCumulativeLinearNNSearch()
    {

    }

    /**
     * Sets the ranked list of features and the window to conduct the knn search on
     * @param target Instance to classify
     * @param window Window to conduct the search with
     * @param activeFeatureIndices Int array containing the indexes of the ranked features
     * @param upperBound Int containing the upper bound of features currently active (used for hill climbing)
     */
    @Override
    public void initialiseCumulativeSearch(Instance target, Instances window,  int[] activeFeatureIndices,int upperBound)
    {
        this.target = target;
        this.window = window;
        distanceFunction.setInstances(window);
        activeFeatures = activeFeatureIndices;
        featureDistance = new double[upperBound][window.numInstances()];
        numberOfActiveFeatures = upperBound - 1; // - 1 as we want the array index

        // initialise lists
        instanceArray = new Inst[window.numInstances()];
        for(int i = 0; i < window.numInstances();i++)
        {
            Inst newInst = new Inst(i);
            instanceArray[i] = newInst;
            searchList.add(newInst);
        }

        // f = index of index of best feature
        double[] tempMaxDist = new double[upperBound];
        for (int f = 0; f < upperBound; f++)
        {
            // maximum distance which can be added on per iteration to the feature's total distance
            double dist = Math.max(target.valueSparse(f),1 - target.valueSparse(f));
            tempMaxDist[f] = dist * dist;

            for(int i = 0; i < window.numInstances();i++)
            {
                // we don't take care of the class index here, the active features array is assumed to NEVER contain the class index.
                // puts squared distance for a feature f between instance i and the target into array.
                double d = distanceFunction.attributeSqDistance(target,window.instance(i),activeFeatures[f]);//= sqDistance(target,window.instance(i),activeFeatures[f]);
                featureDistance[f][i] = d;
                instanceArray[i].distance += d;
            }
        }

        // the maximum distance which can be removed
        double dist = 0;
        for (int i = 0;i<tempMaxDist.length;i++)
        {
            dist += tempMaxDist[i];
            maxDist[i] = dist;
        }



    }

    /**
     * Sets the number of features to be considered in kNN search.
     * Should be run before kNNSearch is run.
     * initialiseCumulativeSearch should be run before running this method
     * backward elimination is assumed
     * @param n number of features to consider
     */
    @Override
    public void setNumberOfActiveFeatures(int n)
    {

        nextIteration();

        /*
        for (int w = 0; w < window.numInstances();w++)
        {
            instanceArray[w].distance = 0;
            for(int f = 0; f < n; f++)
            {
                //System.out.println("f distance: " + featureDistance[f][w]);
                instanceArray[w].distance += featureDistance[f][w];
            }
        }*/
    }

    public boolean nextIteration()
    {
        // didnt work as every feature has been explored
        if(numberOfActiveFeatures == -1)
            return false;

        for(int i = 0; i < searchList.size();i++)
        {
            Inst inst = searchList.get(i);
            inst.distance -= featureDistance[numberOfActiveFeatures][inst.index];
        }
        // list of inst to be searched in this knn search
        Iterator<Inst> it = distanceUpdateList.iterator();
        while (it.hasNext())
        {
            Inst i = it.next();
            i.distance -= featureDistance[numberOfActiveFeatures][i.index];
            if(previousPivotDist != -1)
            {
                if (i.distance - itterationsLeft * maxDist[numberOfActiveFeatures] > previousPivotDist)
                {
                    distanceUpdateList.remove(i);
                }
            }
        }

        numberOfActiveFeatures--;
        return true;
    }


    /**
     * Actual knn search
     * initialise must be called beforehand
     * @param target Instances to search
     * @param kNN Number of nearest neighbours
     * @return A Instances of the k nearest neighbours
     */
    @Override
    public Instances kNNSearch(Instance target,int kNN)
    {
        if(window.numInstances() < kNN)
        {
            //System.out.println("returned window");
            // less instances in window than k
            return window;
        }

        // list of inst to be searched in this knn search
        Iterator<Inst> it = distanceUpdateList.iterator();
        while (it.hasNext())
        {
            Inst i = it.next();
            i.skipCount--;
            if(i.skipCount <= 0)
            {
                searchList.add(i);
                distanceUpdateList.remove(i);
            }

        }


        Instances neighbours = new Instances(window,1);
        // find index of k-th smallest value
        int pivot = kthSmallestValueIndex(searchList, kNN);
        Inst pivotInst = searchList.get(pivot);
        it = searchList.iterator();
        while (it.hasNext())
        {
            Inst i = it.next();
            if(i.distance <= pivotInst.distance)
            {
                neighbours.add(window.instance(i.index));
            }
            else
            {
                int skip = (int)(Math.ceil(i.distance - pivotInst.distance) - 1);
                if (skip > 0)
                {
                    searchList.remove(i);
                    i.skipCount = skip;
                    distanceUpdateList.add(i);
                }
            }
        }

        return neighbours;
    }

    protected int kthSmallestValueIndex(List<Inst> list, int k)
    {
        double[] searchArray = new double[list.size()];
        for(int i = 0; i < list.size();i++)
        {
            searchArray[i] = list.get(i).distance;
        }
        return Utils.kthSmallestValueIndex(searchArray,k);
    }
}