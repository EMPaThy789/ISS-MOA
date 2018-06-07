package moa.classifiers.iss;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Date created: 6/06/2018
 *
 * @author FRANKIE
 */
public class ISSUtils
{
    /**
     * initialises the buffered writer for the output of the dump file
     * only does so if a file name was specified
     * @param fileName
     * @param streamAttributeCount number of attributes in the stream
     * @param featureSearchCount the number of ranked features use for subset selection
     * @return a buffered writer object if the filename was specified, null otherwise
     */
    public static BufferedWriter createDumpFileWriter(String fileName, int streamAttributeCount, int featureSearchCount)
    {
        BufferedWriter bw = null;
        // accuracy gain dump initialisation
        // Only dump if filename is specified
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
                for (int i = 0; i < streamAttributeCount; i++)
                {
                    bw.write("Prediction Accuracy for subset of size" + (i + 1) + ",");
                }
                for (int i = 0; i < streamAttributeCount; i++)
                {
                    bw.write("Accuracy gain for subset of size " + (i + 1) + ",");
                }
                bw.write("Predicted number of relevant features out of total features,");

                bw.write(featureSearchCount + " Number of best ranked features considered");
                bw.write(System.lineSeparator());

            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
        return bw;
    }

    /**
     * writes to dump file for accuracy gain.
     * @param bw buffered writer of dump file
     * @param bestSubsetIndex index of the best subset size (size = index + 1)
     * @param numAttributes number of features in stream
     * @param accuracyEstimateArray accuracy estimate of each subset size (subset of size index + 1)
     * @param accuracyDiffArray accuracy differences between subset from the previous subset (previous subset size = current subset size +1)
     */
    public static void writeAG(BufferedWriter bw, int bestSubsetIndex,int numAttributes, double[] accuracyEstimateArray,double[] accuracyDiffArray)
    {
        // Write ag to file if specified
        if (bw != null)
        {
            try
            {
                // write accuracy estimate
                for (int i = 0; i < accuracyEstimateArray.length; i++)
                {
                    bw.write(Double.toString(accuracyEstimateArray[i]) + ",");

                }

                // wirte accuracy difference
                for (int i = 0; i < accuracyDiffArray.length; i++)
                {
                    bw.write(Double.toString(accuracyDiffArray[i]) + ",");
                }
                bw.write(bestSubsetIndex + " out of " + numAttributes);
                bw.write(System.lineSeparator());
                bw.flush();
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
        }
    }


}
