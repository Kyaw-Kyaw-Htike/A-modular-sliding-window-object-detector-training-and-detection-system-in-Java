// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.StdLib.Matkc;
import KKH.StdLib.stdfuncs;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SimpleLogistic;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.List;

public class classifier_adaboost_weka extends classifier_Base{

    private Classifier classifier;
    private boolean is_trained = false;
    private boolean is_loaded = false;
    private int num_maxIters_train;

    Instances dataset_skeleton;

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
    public double classify(Matkc featVec)
    {
        try{
            Instance data_point = new DenseInstance(1, featVec.vectorize_to_doubleArray());
            data_point.setDataset(dataset_skeleton);
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
    public void train(List<Matkc> feats, List<Integer> labels)
    {
        if(feats.size() != labels.size())
            throw new IllegalArgumentException("ERROR: feats.ncols() != labels.ncols()");

        Utils_det.Results_wekaDataset res_wekaDataset = Utils_det.build_weka_binary_classification_dataset(feats, labels, true);
        Instances dataset_weka = res_wekaDataset.dataset;
        dataset_skeleton = res_wekaDataset.dataset_skeleton;

        SimpleLogistic classifier_temp = new SimpleLogistic();
        //SGD classifier_temp = new SGD();

        try{
            String[] options_classifier = Utils.splitOptions("-M 500 -A -P"); // for SimpleLogistic
            //String[] options_classifier = Utils.splitOptions("-F 0 -C 0.01"); // for SVM
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

    }

    // save the classifier
    public void save(String fpath)
    {
        stdfuncs.serialize_save(fpath, new Object[]{classifier, dataset_skeleton});
    }

    // load the classifier
    public void load(String fpath)
    {
        Object[] objs_loaded = (Object[]) stdfuncs.serialize_load(fpath);
        classifier = (Classifier) objs_loaded[0];
        dataset_skeleton = (Instances) objs_loaded[1];
        is_loaded = true;
    }

    public boolean is_loaded_or_trained()
    {
        return is_trained || is_loaded;
    }
}
