package moa.streams.generators;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.TreeSet;

import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

/**
 *
 * @author Jean Paul Barddal
 */
public class BG extends AbstractOptionHandler implements
        InstanceStream{

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'b', "Balance the number of instances of each class.");

    public IntOption noisePercentageOption = new IntOption("noisePercentage",
            'n', "Percentage of noise to add to the data.", 10, 0, 100);

    public IntOption numFeaturesOption
            = new IntOption("numFeatures", 'F', "", 0, 0, 1024);

    public StringOption relevantFeaturesOption
            = new StringOption("relevantFeatures", 'f', "", "");

    public IntOption numRedundantFeaturesOption
            = new IntOption("numRedundantFeatures", 'R', "", 0, 0, 1000);

    public FloatOption redundancyNoiseProbabilityOption
            = new FloatOption("redundancyNoiseProbability", 'w', "",
                    0.1, 0.0, 1.0);

    protected InstancesHeader streamHeader;

    protected Random instanceRandom;

    protected boolean nextClassShouldBeFalse;

    protected HashMap<Attribute, Attribute> redundantTo;

    protected HashSet<String> namesRelevants;

    FastVector values;
    FastVector classLabels;

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
        // generate header
        FastVector attributes = new FastVector();
        this.values = new FastVector();
        values.add("T");
        values.add("F");

        HashSet<Integer> indicesRelevants = new HashSet<>();
        namesRelevants = new HashSet<>();
        String indices[] = this.relevantFeaturesOption.getValue().split(";");
        for (String strIndex : indices) {
            if (!strIndex.equals("")) {
                int index = Integer.parseInt(strIndex);
                indicesRelevants.add(index);
            }
        }

        for (int i = 0; i < numFeaturesOption.getValue(); i++) {
            attributes.add(new Attribute(("attrib" + (i)), values));
            if (indicesRelevants.contains(i)) {
                namesRelevants.add(("attrib" + (i)));
            }
        }

        this.redundantTo = new HashMap<>();
        this.instanceRandom = new Random(System.currentTimeMillis());

        int priorFeaturesAmount = attributes.size();

        int nextToBeCopied = 0;
        for (int i = 0; i < this.numRedundantFeaturesOption.getValue(); i++) {

            //picks a random feature to be used as source
            Attribute original = (Attribute) attributes.get(nextToBeCopied % priorFeaturesAmount);
            nextToBeCopied++;

            ArrayList<String> auxvalues = new ArrayList<>();

            for (Enumeration<String> enumeration = original.enumerateValues(); enumeration.hasMoreElements();) {
                auxvalues.add(enumeration.nextElement());
            }
            Attribute redundant = new Attribute("attrib" + (attributes.size()), auxvalues);
            attributes.addElement(redundant);

            redundantTo.put(redundant, original);
        }

        this.classLabels = new FastVector();
        classLabels.addElement("groupA");
        classLabels.addElement("groupB");
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);

        restart();
    }

    @Override
    public long estimatedRemainingInstances() {
        return -1;
    }

    @Override
    public InstancesHeader getHeader() {
        return this.streamHeader;
    }

    @Override
    public boolean hasMoreInstances() {
        return true;
    }

    @Override
    public boolean isRestartable() {
        return true;
    }

    @Override
    public InstanceExample nextInstance() {
        boolean classValue = nextClassShouldBeFalse == false ? true : false;

        if (balanceClassesOption.isSet()) {
            nextClassShouldBeFalse = !nextClassShouldBeFalse;
        } else {
            nextClassShouldBeFalse = this.instanceRandom.nextBoolean();
        }

        // generate attributes
        boolean atts[] = createInstance(classValue);

        //Add Noise
        if ((1 + (this.instanceRandom.nextInt(100)))
                <= this.noisePercentageOption.getValue()) {
            classValue = (classValue == false ? true : false);
        }

        // construct instance
        InstancesHeader header = getHeader();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);
        for (int i = 0; i < atts.length; i++) {
//            int value = atts[i] ? 1 : 0;
            String value = atts[i] ? "T" : "F";
            inst.setValue(i, indexOfValue(value, values.toArray()));
        }

        //buils redundant features
        for (Attribute att : this.redundantTo.keySet()) {
            Attribute baseCopy = redundantTo.get(att);
            int indexAttCopied = (int) indexOfAttribute(inst, baseCopy);
            int valueAttCopied = (int) inst.value(indexOfAttribute(inst, baseCopy));
            int indexAttTarget = (int) indexOfAttribute(inst, att);
            inst.setValue(indexAttTarget, valueAttCopied);
            int valueAttTarget = (int) inst.value(indexAttTarget);

            if (this.instanceRandom.nextDouble() <= this.redundancyNoiseProbabilityOption.getValue()) {
                // picks a random value of the feature,
                // wih the exception of the original one
                int index = this.instanceRandom.nextInt(att.numValues());
                while (baseCopy.value(index).equals(inst.value(indexAttCopied))) {
                    index = this.instanceRandom.nextInt(att.numValues());
                }

                //sets the value                             
                inst.setValue(indexAttTarget, index);
            }
        }

        inst.setClassValue(classValue ? indexOfValue("groupA", classLabels.toArray()) : indexOfValue("groupB", classLabels.toArray()));
//        System.out.println(inst);
        return new InstanceExample(inst);
//        return (Example<Instance>) inst;
    }

    @Override
    public void restart() {
        this.instanceRandom
                = new Random(this.instanceRandomSeedOption.getValue());
        this.nextClassShouldBeFalse = false;
        //Outputs all relevant attributes' names
        System.out.print("relevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if (namesRelevants.contains(att.name())
                    && !this.redundantTo.keySet().contains(att)) {
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \t");

        // irrelevant
        System.out.print("irrelevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if (!namesRelevants.contains(att.name())
                    && !this.redundantTo.keySet().contains(att)) {
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \t");

        //redundant
        System.out.print("redundant = [");
        for (Attribute att : this.redundantTo.keySet()) {
            System.out.print(att.name() + ",");
        }
        System.out.print("] \t");

        //redundant to
        System.out.print("redundant to = [");
        for (Attribute att : this.redundantTo.keySet()) {
            System.out.print(att.name() + "<->" + redundantTo.get(att).name() + ",");
        }
        System.out.print("] \n");

    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    public String getPurposeString() {
        return "Generates a AND concept accordingly to binary variables.";
    }

    private boolean[] createInstance(boolean classValue) {
        boolean atts[] = new boolean[numFeaturesOption.getValue()];
        for (int i = 0; i < atts.length; i++) {
            atts[i] = this.instanceRandom.nextBoolean();
        }
        if (classValue == true) {
            String indices[] = this.relevantFeaturesOption.getValue().split(";");
            for (String strIndex : indices) {
                if (!strIndex.equals("")) {
                    int index = Integer.parseInt(strIndex);
                    atts[index] = true;
                }
            }
        } else {
//            //at least one of the attributes must be set to false
//            //therefore, we randomize which attributes will be set to false
            String indices[] = this.relevantFeaturesOption.getValue().split(";");
            boolean obtainedClassValue = false;
            while (!obtainedClassValue) {
                for (String strIndex : indices) {
                    if (!strIndex.equals("")) {
                        int index = Integer.parseInt(strIndex);
                        if (this.instanceRandom.nextDouble() < 0.5) {
                            atts[index] = false;
                        }
                    }
                }

                for (String strIndex : indices) {
                    if (!strIndex.equals("")) {
                        int index = Integer.parseInt(strIndex);
                        if (!atts[index]) {
                            obtainedClassValue = true;
                        }
                    }
                }
            }

        }

        return atts;
    }

    private static int indexOfValue(String value, Object[] arr) {
        int index = Arrays.asList(arr).indexOf(value);
        return index;
    }

    private int indexOfAttribute(Instance instnc, Attribute att) {
        for (int i = 0; i < instnc.numAttributes(); i++) {
            if (instnc.attribute(i).name().equals(att.name())) {
                return i;
            }
        }
        return -1;
    }

}
