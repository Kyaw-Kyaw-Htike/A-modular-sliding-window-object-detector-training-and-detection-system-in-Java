// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.StdLib.Matkc;
import KKH.StdLib.stdfuncs;
import jsat.classifiers.CategoricalResults;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.boosting.AdaBoostM1;
import jsat.classifiers.trees.DecisionTree;
import jsat.classifiers.trees.RandomForest;
import jsat.linear.DenseVector;

import java.util.List;

public class classifier_adaboost_jsat extends classifier_Base{

    private Classifier classifier;
    private boolean is_trained = false;
    private boolean is_loaded = false;
    private int num_maxIters_train;

    public classifier_adaboost_jsat()
    {
        num_maxIters_train = 100;
    }

    public classifier_adaboost_jsat(int num_maxIters)
    {
        num_maxIters_train = num_maxIters;
    }

    /**
     * Classify a given feature vector
     * @param featVec should be a Matkc column vector
     * @return classification score
     */
    @Override
    public double classify(Matkc featVec)
    {
        CategoricalResults res = classifier.classify(new DataPoint(new DenseVector(featVec.vectorize_to_doubleArray())));
        return res.getProb(0);
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

        ClassificationDataSet dataset_jsat = Utils_det.build_jsat_binary_classification_dataset(feats, labels, true);

        //DecisionTree cobj_weak = new DecisionTree(2);
        //DecisionTree cobj_weak = DecisionTree.getC45Tree();
        //cobj_weak.setMaxDepth(3);
//        DecisionStump cobj_weak = new DecisionStump();
        //classifier = new AdaBoostM1(cobj_weak, num_maxIters_train);

        RandomForest rf_classifier = new RandomForest(1000);
        rf_classifier.autoFeatureSample();
        classifier = rf_classifier;

        System.out.println("Training the AdaBoost classifier...");
        classifier.trainC(dataset_jsat);
        System.out.println("Classifier trained.");
        is_trained = true;
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
