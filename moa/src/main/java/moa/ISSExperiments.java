package moa;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.bayes.NaiveBayesISS;
import moa.classifiers.lazy.kNN;
import moa.classifiers.lazy.kNNISS;
import moa.evaluation.LearningCurve;
import moa.streams.ConceptDriftStream;
import moa.streams.ExampleStream;
import moa.streams.generators.AgrawalGenerator;
import moa.streams.generators.AssetNegotiationGenerator;
import moa.streams.generators.BG;
import moa.streams.generators.BG2;
import moa.streams.generators.BG3;
import moa.streams.generators.ConditionalGenerator;
import moa.streams.generators.LEDGeneratorDrift;
import moa.streams.generators.SEAFD;
import moa.tasks.EvaluatePrequential;

/**
 *
 * @author Jean Paul Barddal
 */
public class ISSExperiments {

    private static int NUM_INSTANCES_STREAM = 500000;
    private static int EVALUATION_FREQUENCY = 10000;
    private static int DRIFT_POSITION_SINGLE_DRIFT = 250000;
    private static int DRIFT_POSITION_TWO_DRIFTS = 200000;

    public static void main(String args[]) throws FileNotFoundException, UnsupportedEncodingException {

        // Instantiates all classifiers and hyper-parameter combinations
        HashMap<String, Classifier> classifiers = new HashMap<>();
        classifiers.putAll(instantiateNaiveBayes());
//        classifiers.putAll(instantiateKNN());

        // Instantiates all the synthetic experiments
        HashMap<String, ExampleStream> streams = new HashMap<>();
        streams.putAll(instantiateAGRAWAL()); // AGRAWAL
        streams.putAll(instantiateASSETS()); // ASSETS
        streams.putAll(instantiateBG1()); // BG1
        streams.putAll(instantiateBG2()); // BG2
        streams.putAll(instantiateBG3()); // BG3
        streams.putAll(instantiateConditionalGenerator()); // CONDITIONAL GENERATOR
        streams.putAll(instantiateFY()); // FY DATASETS
        streams.putAll(instantiateLEDDrift()); // LED w/ DRIFT
        streams.putAll(instantiateSEA()); // SEA w/ FEATURE DRIFT

        // Prepares the folder that will contain all the results
        prepareFolder();

        // Prepares a buffer to output the summary file
        PrintWriter writer = new PrintWriter(new FileOutputStream("./summary.csv", false));
        writer.println("Stream,Classifier,Avg Accuracy, CPU Time, RAM-Hours");

        // Loops over all streams and classifiers to run the experiments
        for (String strStream : streams.keySet()) {
            ExampleStream s = streams.get(strStream);
            for (String strClassifier : classifiers.keySet()) {
                Classifier c = classifiers.get(strClassifier);
                appendResults(writer, runExperiment(strClassifier, c, strStream, s));
            }
        }
        writer.close();
    }

    private static String runExperiment(String strClassifier, Classifier c,
            String strStream, ExampleStream s) {
        // prepares the stream and the classifier for execution
        c.prepareForUse();
        c.resetLearning();
        s.restart();

        // prepares the filename for outputting raw results
        String filename = prepareFileName(strClassifier, strStream);

        // runs the experiment
        EvaluatePrequential prequential = new EvaluatePrequential();
        prequential.prepareForUse();
        prequential.instanceLimitOption.setValue(NUM_INSTANCES_STREAM);
        prequential.sampleFrequencyOption.setValue(EVALUATION_FREQUENCY);
        prequential.dumpFileOption.setValue("./results/" + filename);
        prequential.streamOption.setCurrentObject(s);
        prequential.learnerOption.setCurrentObject(c);
        LearningCurve lc = (LearningCurve) prequential.doTask();

        // extracts the final results for summary
        double rs[] = getFinalValuesForExperiment(lc);
        double avgAccuracy = rs[0];
        double cpuTime = rs[1];
        double ramHours = rs[2];

        // RETURNS (STREAM, CLASSIFIER, AVG_ACCURACY, CPU_TIME, RAM_HOURS)        
        return strStream + "," + strClassifier + "," + avgAccuracy + ","
                + cpuTime + "," + ramHours;

    }

    private static void appendResults(PrintWriter writer, String line) {
        writer.println(line);
        writer.flush();
    }

    private static HashMap<String, Classifier> instantiateNaiveBayes()
    {
        HashMap<String, Classifier> classifiers = new HashMap<>();
        classifiers.put("NB", new NaiveBayes());
        //int rankingOptionI = 0
        for(int rankingOptionI = 0; rankingOptionI < 3; rankingOptionI++)
        {

            for (int windowSizeI = 1; windowSizeI <= 3; windowSizeI++)
            {

                for (int reselectionI = 1; reselectionI <= 3; reselectionI++)
                {
                    NaiveBayesISS nbISS = new NaiveBayesISS();
                    int reselectionInterval = 1000 * reselectionI;
                    int windowSize = 500 * windowSizeI;
                    nbISS.rankingWindowSizeOption.setValue(windowSize);
                    nbISS.reselectionIntervalOption.setValue(reselectionInterval);
                    nbISS.rankingOption.setChosenIndex(rankingOptionI);
                    String rankingFunc = "SU";

                    if (rankingOptionI == 1)
                    {
                        rankingFunc = "IG";
                    }
                    if (rankingOptionI == 2)
                    {
                        rankingFunc = "AED";
                    }
                    classifiers.put("NB-ISS-" + rankingFunc + " (window: " + nbISS.rankingWindowSizeOption.getValue() + " reselectionInterval:  " + nbISS.reselectionIntervalOption.getValue() + " decay: " + nbISS.decayFactorOption.getValue() + " decayInterval: " + nbISS.decayIntervalOption.getValue(), nbISS);
                }
            }

            for (int decayIntervalI = 1; decayIntervalI <= 3; decayIntervalI++) {
                for (int decayI = 1; decayI < 4; decayI++)
                {

                    NaiveBayesISS nbISS = new NaiveBayesISS();
                    double decay = (double) (decayI) * 0.05;
                    int decayInterval = 1000 * decayIntervalI;
                    nbISS.decayFactorOption.setValue(decay);
                    nbISS.decayIntervalOption.setValue(decayInterval);
                    nbISS.rankingOption.setChosenIndex(rankingOptionI);
                    String rankingFunc = "SU";

                    if (rankingOptionI == 1)
                    {
                        rankingFunc = "IG";
                    }
                    if (rankingOptionI == 2)
                    {
                        rankingFunc = "AED";
                    }
                    classifiers.put("NB-ISS-" + rankingFunc + " (window: " + nbISS.rankingWindowSizeOption.getValue() + " reselectionInterval:  " + nbISS.reselectionIntervalOption.getValue() + " decay: " + nbISS.decayFactorOption.getValue() + " decayInterval: " + nbISS.decayIntervalOption.getValue(), nbISS);

                }
            }
        }
        return classifiers;
    }

    private static HashMap<String, Classifier> instantiatekNNHC()
    {
        HashMap<String, Classifier> classifiers = new HashMap<>();
        classifiers.put("KNN", new kNN());
        int rankingOptionI = 0;

        //for(int rankingOptionI = 0; rankingOptionI < 3; rankingOptionI++)
        {
            // window, reselection and hc
            for (int windowSizeI = 1; windowSizeI <= 3; windowSizeI++)
            {
                for (int reselectionI = 1; reselectionI <= 3; reselectionI++)
                {
                    for (int hillClimbWindowI = 0; hillClimbWindowI < 3; hillClimbWindowI++)
                    {
                        kNNISS kNNISSClassifier = new kNNISS();
                        int reselectionInterval = 500 * reselectionI;
                        int windowSize = 500 * windowSizeI;
                        int hillClimbWindowSize = 2 * (hillClimbWindowI + 1);
                        String rankingFunc = "SU";


                        kNNISSClassifier.hillClimbOption.setValue(true);

                        kNNISSClassifier.hillClimbWindowOption.setValue(hillClimbWindowSize);
                        kNNISSClassifier.windowSizeOption.setValue(windowSize);
                        kNNISSClassifier.reselectionIntervalOption.setValue(reselectionInterval);
                        kNNISSClassifier.rankingOption.setChosenIndex(rankingOptionI);

                        if (rankingOptionI == 1) {
                            rankingFunc = "IG";
                        }
                        if (rankingOptionI == 2) {
                            rankingFunc = "AED";
                        }

                        classifiers.put("kNN-ISS-HC-" + rankingFunc + " (window: " + kNNISSClassifier.windowSizeOption.getValue() + " reselectionInterval:  " + kNNISSClassifier.reselectionIntervalOption.getValue() + " decay: " + kNNISSClassifier.decayFactorOption.getValue() + " decayInterval: " + kNNISSClassifier.decayIntervalOption.getValue() + " hillClimbWindow: " + kNNISSClassifier.hillClimbWindowOption.getValue(), kNNISSClassifier);
                    }
                }
            }

            // decay
            for (int decayIntervalI = 1; decayIntervalI <= 3; decayIntervalI++)
            {
                for (int decayI = 1; decayI < 8; decayI++)
                {

                    kNNISS kNNISSClassifier = new kNNISS();
                    double decay = (double)(decayI) * 0.05;
                    int decayInterval = 500 * decayIntervalI;
                    kNNISSClassifier.decayFactorOption.setValue(decay);
                    kNNISSClassifier.decayIntervalOption.setValue(decayInterval);
                    String rankingFunc = "SU";
                    if (rankingOptionI == 1) {
                        rankingFunc = "IG";
                    }
                    if (rankingOptionI == 2) {
                        rankingFunc = "AED";
                    }
                    classifiers.put("kNN-ISS-HC-" + rankingFunc + " (window: " + kNNISSClassifier.windowSizeOption.getValue() + " reselectionInterval:  " + kNNISSClassifier.reselectionIntervalOption.getValue() + " decay: " + kNNISSClassifier.decayFactorOption.getValue() + " decayInterval: " + kNNISSClassifier.decayIntervalOption.getValue() + " hillClimbWindow: " + kNNISSClassifier.hillClimbWindowOption.getValue(), kNNISSClassifier);
                }
            }
        }
        return classifiers;
    }

    private static HashMap<String, Classifier> instantiateKNN() {
        HashMap<String, Classifier> classifiers = new HashMap<>();
        classifiers.put("KNN", new kNN());
        classifiers.put("KNN-ISS", new kNNISS());
        return classifiers;
    }

    private static HashMap<String, ExampleStream> instantiateAGRAWAL() {
        HashMap<String, ExampleStream> streams = new HashMap<>();

        AgrawalGenerator pt1 = new AgrawalGenerator();
        pt1.balanceClassesOption.set();
        pt1.prepareForUse();

        AgrawalGenerator pt2 = new AgrawalGenerator();
        pt2.balanceClassesOption.set();
        pt2.functionOption.setValue(4);
        pt2.prepareForUse();

        AgrawalGenerator pt3 = new AgrawalGenerator();
        pt3.balanceClassesOption.set();
        pt3.functionOption.setValue(8);
        pt3.prepareForUse();

        // Creates the final stream
        ConceptDriftStream str = new ConceptDriftStream();
        str.streamOption.setCurrentObject(pt1);
        str.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);

        ConceptDriftStream str2 = new ConceptDriftStream();

        str2.streamOption.setCurrentObject(pt2);
        str2.driftstreamOption.setCurrentObject(pt3);
        str2.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        str2.prepareForUse();

        str.driftstreamOption.setCurrentObject(str2);
        str.prepareForUse();

        streams.put("AGRAWAL(GRADUAL)", str);
        return streams;
    }

    private static HashMap<String, ExampleStream> instantiateASSETS() {
        HashMap<String, ExampleStream> streams = new HashMap<>();

        AssetNegotiationGenerator pt1 = new AssetNegotiationGenerator();
        pt1.prepareForUse();

        AssetNegotiationGenerator pt2 = new AssetNegotiationGenerator();
        pt2.functionOption.setValue(3);
        pt2.prepareForUse();

        AssetNegotiationGenerator pt3 = new AssetNegotiationGenerator();
        pt3.functionOption.setValue(2);
        pt3.prepareForUse();

        // Creates the final stream
        ConceptDriftStream str = new ConceptDriftStream();
        str.streamOption.setCurrentObject(pt1);
        str.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);

        ConceptDriftStream str2 = new ConceptDriftStream();

        str2.streamOption.setCurrentObject(pt2);
        str2.driftstreamOption.setCurrentObject(pt3);
        str2.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        str2.prepareForUse();

        str.driftstreamOption.setCurrentObject(str2);
        str.prepareForUse();

        streams.put("ASSETS(GRADUAL)", str);
        return streams;
    }

    private static HashMap<String, ExampleStream> instantiateBG1() {
        HashMap<String, ExampleStream> streams = new HashMap<>();

        BG pt1 = new BG();
        pt1.relevantFeaturesOption.setValue("0;1;2");
        pt1.numFeaturesOption.setValue(10);
        pt1.balanceClassesOption.set();
        pt1.prepareForUse();

        BG pt2 = new BG();
        pt2.relevantFeaturesOption.setValue("4;5;6");
        pt2.numFeaturesOption.setValue(10);
        pt2.balanceClassesOption.set();
        pt2.prepareForUse();

        BG pt3 = new BG();
        pt3.relevantFeaturesOption.setValue("0;1;4");
        pt3.numFeaturesOption.setValue(10);
        pt3.balanceClassesOption.set();
        pt3.prepareForUse();

        // Creates the final stream
        ConceptDriftStream str = new ConceptDriftStream();
        str.streamOption.setCurrentObject(pt1);
        str.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);

        ConceptDriftStream str2 = new ConceptDriftStream();

        str2.streamOption.setCurrentObject(pt2);
        str2.driftstreamOption.setCurrentObject(pt3);
        str2.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        str2.prepareForUse();

        str.driftstreamOption.setCurrentObject(str2);
        str.prepareForUse();

        streams.put("BG(GRADUAL)", str);

        return streams;
    }

    private static HashMap<String, ExampleStream> instantiateBG2() {
        HashMap<String, ExampleStream> streams = new HashMap<>();

        BG2 pt1 = new BG2();
        pt1.instanceRandomSeedOption.setValue(463517);
        pt1.numIrrelevantFeaturesOption.setValue(15);
        pt1.balanceClassesOption.set();
        pt1.prepareForUse();

        BG2 pt2 = new BG2();        
        pt2.instanceRandomSeedOption.setValue(3871623);
        pt2.numIrrelevantFeaturesOption.setValue(15);        
        pt2.balanceClassesOption.set();
        pt2.prepareForUse();

        BG2 pt3 = new BG2();
        pt3.instanceRandomSeedOption.setValue(897341);
        pt3.numIrrelevantFeaturesOption.setValue(15);       
        pt3.balanceClassesOption.set();
        pt3.prepareForUse();

        // Creates the final stream
        ConceptDriftStream str = new ConceptDriftStream();
        str.streamOption.setCurrentObject(pt1);
        str.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);

        ConceptDriftStream str2 = new ConceptDriftStream();

        str2.streamOption.setCurrentObject(pt2);
        str2.driftstreamOption.setCurrentObject(pt3);
        str2.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        str2.prepareForUse();

        str.driftstreamOption.setCurrentObject(str2);
        str.prepareForUse();

        streams.put("BG 2 (GRADUAL)", str);
        return streams;
    }

    private static HashMap<String, ExampleStream> instantiateBG3() {
        HashMap<String, ExampleStream> streams = new HashMap<>();
        
        BG3 pt1 = new BG3();
        pt1.instanceRandomSeedOption.setValue(463517);
        pt1.numIrrelevantFeaturesOption.setValue(15);
        pt1.balanceClassesOption.set();
        pt1.prepareForUse();

        BG3 pt2 = new BG3();        
        pt2.instanceRandomSeedOption.setValue(3871623);
        pt2.numIrrelevantFeaturesOption.setValue(15);        
        pt2.balanceClassesOption.set();
        pt2.prepareForUse();

        BG3 pt3 = new BG3();
        pt3.instanceRandomSeedOption.setValue(897341);
        pt3.numIrrelevantFeaturesOption.setValue(15);       
        pt3.balanceClassesOption.set();
        pt3.prepareForUse();

        // Creates the final stream
        ConceptDriftStream str = new ConceptDriftStream();
        str.streamOption.setCurrentObject(pt1);
        str.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);

        ConceptDriftStream str2 = new ConceptDriftStream();

        str2.streamOption.setCurrentObject(pt2);
        str2.driftstreamOption.setCurrentObject(pt3);
        str2.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        str2.prepareForUse();

        str.driftstreamOption.setCurrentObject(str2);
        str.prepareForUse();

        streams.put("BG 3 (GRADUAL)", str);
        
        return streams;
    }

    private static HashMap<String, ExampleStream> instantiateConditionalGenerator() {
        HashMap<String, ExampleStream> streams = new HashMap<>();
        ConditionalGenerator conditional = new ConditionalGenerator();
        conditional.driftIntervalOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        conditional.driftOption.set();
        conditional.irrelevantNominalOption.setValue(10);
        conditional.irrelevantNumericOption.setValue(10);
        conditional.nomOption.setValue(10);
        conditional.numOption.setValue(10);
        conditional.prepareForUse();
        streams.put("CONDITIONAL(2 drifts)", conditional);
        return streams;
    }


    private static HashMap<String, ExampleStream> instantiateFY()
    {
        HashMap<String, ExampleStream> streams = new HashMap<>();

        ConditionalGenerator FYA = new ConditionalGenerator();
        FYA.seedOption.setValue(1);
        FYA.relevantNominalOption.setValue(10);
        FYA.relevantNumericOption.setValue(10);
        FYA.irrelevantNominalOption.setValue(10);
        FYA.irrelevantNumericOption.setValue(10);
        FYA.nomOption.setValue(10);
        FYA.numOption.setValue(10);
        FYA.classOption.setValue(4);
        FYA.nomOption.setValue(10);
        FYA.numOption.setValue(5);
        FYA.noisePercentageOption.setValue(10);
        FYA.prepareForUse();
        streams.put("FY_A", FYA);

        ConditionalGenerator FYB = new ConditionalGenerator();
        FYB.seedOption.setValue(1);
        FYB.relevantNominalOption.setValue(10);
        FYB.relevantNumericOption.setValue(10);
        FYB.irrelevantNominalOption.setValue(50);
        FYB.irrelevantNumericOption.setValue(50);
        FYB.nomOption.setValue(10);
        FYB.numOption.setValue(10);
        FYB.classOption.setValue(4);
        FYB.nomOption.setValue(10);
        FYB.numOption.setValue(5);
        FYB.noisePercentageOption.setValue(10);
        FYB.prepareForUse();
        streams.put("FY_B", FYB);

        ConditionalGenerator FYC = new ConditionalGenerator();
        FYC.seedOption.setValue(1);
        FYC.driftIntervalOption.setValue(DRIFT_POSITION_SINGLE_DRIFT);
        FYC.driftOption.set();
        FYC.relevantNominalOption.setValue(3);
        FYC.relevantNumericOption.setValue(3);
        FYC.irrelevantNominalOption.setValue(50);
        FYC.irrelevantNumericOption.setValue(50);
        FYC.nomOption.setValue(10);
        FYC.numOption.setValue(10);
        FYC.classOption.setValue(10);
        FYC.nomOption.setValue(25);
        FYC.numOption.setValue(25);
        FYC.noisePercentageOption.setValue(10);
        FYC.prepareForUse();
        streams.put("FY_C", FYC);

        ConditionalGenerator FYD = new ConditionalGenerator();
        FYD.seedOption.setValue(1);
        FYD.driftIntervalOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        FYD.driftOption.set();
        FYD.relevantNominalOption.setValue(10);
        FYD.relevantNumericOption.setValue(10);
        FYD.irrelevantNominalOption.setValue(50);
        FYD.irrelevantNumericOption.setValue(50);
        FYD.nomOption.setValue(10);
        FYD.numOption.setValue(10);
        FYD.classOption.setValue(10);
        FYD.nomOption.setValue(25);
        FYD.numOption.setValue(25);
        FYD.noisePercentageOption.setValue(10);
        FYD.prepareForUse();
        streams.put("FY_D", FYD);

        ConditionalGenerator FYADrift = new ConditionalGenerator();
        FYADrift.seedOption.setValue(1);
        FYADrift.driftIntervalOption.setValue(DRIFT_POSITION_SINGLE_DRIFT);
        FYADrift.relevantNominalOption.setValue(10);
        FYADrift.relevantNumericOption.setValue(10);
        FYADrift.irrelevantNominalOption.setValue(10);
        FYADrift.irrelevantNumericOption.setValue(10);
        FYADrift.nomOption.setValue(10);
        FYADrift.numOption.setValue(10);
        FYADrift.classOption.setValue(4);
        FYADrift.nomOption.setValue(10);
        FYADrift.numOption.setValue(5);
        FYADrift.noisePercentageOption.setValue(10);
        FYADrift.prepareForUse();
        streams.put("FY_A_Drift", FYADrift);

        ConditionalGenerator FYBDrift = new ConditionalGenerator();
        FYBDrift.seedOption.setValue(1);
        FYBDrift.driftIntervalOption.setValue(DRIFT_POSITION_SINGLE_DRIFT);
        FYBDrift.relevantNominalOption.setValue(10);
        FYBDrift.relevantNumericOption.setValue(10);
        FYBDrift.irrelevantNominalOption.setValue(50);
        FYBDrift.irrelevantNumericOption.setValue(50);
        FYBDrift.nomOption.setValue(10);
        FYBDrift.numOption.setValue(10);
        FYBDrift.classOption.setValue(4);
        FYBDrift.nomOption.setValue(10);
        FYBDrift.numOption.setValue(5);
        FYBDrift.noisePercentageOption.setValue(10);
        FYBDrift.prepareForUse();
        streams.put("FY_B_Drift", FYBDrift);

        return streams;
    }

    private static HashMap<String, ExampleStream> instantiateLEDDrift() {
        HashMap<String, ExampleStream> streams = new HashMap<>();
        LEDGeneratorDrift pt1 = new LEDGeneratorDrift();
        pt1.numberAttributesDriftOption.setValue(0);
        pt1.prepareForUse();

        LEDGeneratorDrift pt2 = new LEDGeneratorDrift();
        pt2.numberAttributesDriftOption.setValue(4);
        pt2.prepareForUse();

        // Creates the final stream
        ConceptDriftStream str = new ConceptDriftStream();
        str.streamOption.setCurrentObject(pt1);
        str.driftstreamOption.setCurrentObject(pt2);
        str.positionOption.setValue(DRIFT_POSITION_SINGLE_DRIFT);
        str.prepareForUse();

        streams.put("LED(GRADUAL)", str);
        return streams;
    }

    private static HashMap<String, ExampleStream> instantiateSEA() {
        HashMap<String, ExampleStream> streams = new HashMap<>();

        SEAFD pt1 = new SEAFD();
        pt1.balanceClassesOption.set();
        pt1.numRandomAttsOption.setValue(10);
        pt1.prepareForUse();

        SEAFD pt2 = new SEAFD();
        pt2.instanceRandomSeedOption.setValue(813727813);
        pt2.balanceClassesOption.set();
        pt2.numRandomAttsOption.setValue(10);
        pt2.prepareForUse();

        SEAFD pt3 = new SEAFD();
        pt3.instanceRandomSeedOption.setValue(4786123);
        pt3.balanceClassesOption.set();
        pt3.numRandomAttsOption.setValue(10);
        pt3.prepareForUse();

        // Creates the final stream
        ConceptDriftStream str = new ConceptDriftStream();
        str.streamOption.setCurrentObject(pt1);
        str.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);

        ConceptDriftStream str2 = new ConceptDriftStream();

        str2.streamOption.setCurrentObject(pt2);
        str2.driftstreamOption.setCurrentObject(pt3);
        str2.positionOption.setValue(DRIFT_POSITION_TWO_DRIFTS);
        str2.prepareForUse();

        str.driftstreamOption.setCurrentObject(str2);
        str.prepareForUse();

        streams.put("SEA(GRADUAL)", str);

        return streams;
    }



    ////////////////////////////////////////////////////////////////
    // AUXILIAR METHOD TO EXTRACT RESULTS FROM THE LEARNING CURVE //
    ////////////////////////////////////////////////////////////////
    private static double[] getFinalValuesForExperiment(LearningCurve lc) {
        int indexAcc = -1;
        int indexCpuTime = -1;
        int indexRamHours = -1;
        int index = 0;
        for (String s : lc.headerToString().split(",")) {
            if (s.contains("classifications correct")) {
                indexAcc = index;
            } else if (s.contains("time")) {
                indexCpuTime = index;
            } else if (s.contains("RAM-Hours")) {
                indexRamHours = index;
            }
            index++;
        }

        // the accuracy must be averaged
        double avgAccuracy = 0.0;
        for (int entry = 0; entry < lc.numEntries(); entry++) {
            avgAccuracy += lc.getMeasurement(entry, indexAcc);
        }
        avgAccuracy /= lc.numEntries();

        // but both cpu time and ram hours are only the final values obtained
        // since they represent the processing of the entire stream
        double cpuTime = lc.getMeasurement(lc.numEntries() - 1, indexCpuTime);
        double ramHours = lc.getMeasurement(lc.numEntries() - 1, indexRamHours);

        return new double[]{avgAccuracy, cpuTime, ramHours};
    }

    /////////////////////////////////////////////
    // AUXILIAR METHODS TO SET UP OUTPUT FILES //
    /////////////////////////////////////////////
    private static void prepareFolder() {
        File folder = new File("./results/");
        File listOfFiles[];
        if (folder.exists()) {
            listOfFiles = folder.listFiles();
            for (int i = 0; i < listOfFiles.length; i++) {
                if (listOfFiles[i].isFile()) {
                    if (listOfFiles[i].getName().endsWith(".csv")) {
                        listOfFiles[i].delete();
                    }
                }
            }
        } else {
            folder.mkdir();
        }
    }

    private static String prepareFileName(String strClassifier, String strStream) {
        String filename = strStream + "_" + strClassifier + ".csv";
        filename = filename.trim();
        filename = filename.replace("-", "_").replace(" ", "_");
        return filename;
    }

}
