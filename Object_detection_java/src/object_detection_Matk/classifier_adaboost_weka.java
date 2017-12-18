// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.StdLib.Matk;
import KKH.StdLib.stdfuncs;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.boosting.AdaBoostM1;
import jsat.classifiers.trees.DecisionTree;
import jsat.linear.DenseVector;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SimpleLogistic;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.List;

import weka.core.Utils;

/**
 * Created by Kyaw on 21-May-17.
 */
public class classifier_adaboost_weka extends classifier_Base{

    private Classifier classifier;
    private boolean is_trained = false;
    private boolean is_loaded = false;
    private int num_maxIters_train;

    private double[] v_datapoint;
    Instances dataset_train;

    public classifier_adaboost_weka()
    {
        num_maxIters_train = 100;
    }

    public classifier_adaboost_weka(int num_maxIters)
    {
        num_maxIters_train = num_maxIters;
    }

    /**
     * Classify a given feature vector
     * @param featVec should be a Matk column vector
     * @return classification score
     */
    @Override
    public double classify(Matk featVec)
    {
        if(v_datapoint == null) v_datapoint = new double[featVec.ndata()+1];

        try{
            System.arraycopy(featVec.vectorize_to_doubleArray(), 0, v_datapoint, 0, v_datapoint.length-1);
            //v_datapoint[v_datapoint.length-1] = 0;
            Instance data_point = new DenseInstance(1, v_datapoint);
            data_point.setDataset(dataset_train);
            //data_point.setClassMissing();
            double[] res = classifier.distributionForInstance(data_point);
            return res[0];
        }
        catch (Exception e)
        {
            e.printStackTrace();
            throw new IllegalArgumentException("ERROR: could not apply the classifier to a single data point");
        }

    }

    /**
     * Train a classifier
     * @param feats
     * @param labels
     */
    @Override
    public void train(List<Matk> feats, List<Integer> labels)
    {
        if(feats.size() != labels.size())
            throw new IllegalArgumentException("ERROR: feats.ncols() != labels.ncols()");

        Instances dataset_weka = Utils_det.build_weka_binary_classification_dataset(feats, labels, true);

        //SimpleLogistic classifier_temp = new SimpleLogistic();
        SGD classifier_temp = new SGD();
        try{
            //String[] options_classifier = Utils.splitOptions("-M 500 -A -P"); // for SimpleLogistic
            String[] options_classifier = Utils.splitOptions("-F 0 -C 0.01"); // for SVM
            classifier_temp.setOptions(options_classifier);
        }
        catch(Exception e)
        {
            throw new IllegalArgumentException("ERROR: Invalid classifier options for training.");
        }

        System.out.println("Training the Weka classifier...");
        try
        {
            classifier_temp.buildClassifier(dataset_weka);
        }
        catch(Exception e)
        {
            System.out.println("Could not successfully train classifier.");
            e.printStackTrace();
        }

        System.out.println("Classifier trained.");
        is_trained = true;
        classifier = classifier_temp;
        dataset_train = dataset_weka;

    }

    // save the classifier
    public void save(String fpath)
    {
        stdfuncs.serialize_save(fpath, classifier);
    }

    // load the classifier
    public void load(String fpath)
    {
        classifier = (Classifier)stdfuncs.serialize_load(fpath);
        is_loaded = true;
    }

    public boolean is_loaded_or_trained()
    {
        return is_trained || is_loaded;
    }
}
