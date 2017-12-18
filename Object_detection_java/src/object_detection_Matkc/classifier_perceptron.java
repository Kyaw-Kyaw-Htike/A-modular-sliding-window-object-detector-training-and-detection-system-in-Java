// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.StdLib.Matkc;
import com.google.common.primitives.Ints;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class classifier_perceptron extends classifier_Base {

    private Matkc w_lin;
    private double bias;

    /**
     * Classify a given feature vector
     * @param featVec should be a Matk column vector
     * @return classification score
     */
    @Override
    public double classify(Matkc featVec)
    {
        return w_lin.dot(featVec);
    }

    /**
     * Train a classifier
     * @param feats
     * @param labels_
     */
    @Override
    public void train(List<Matkc> feats, List<Integer> labels_)
    {
        if(feats.size() != labels_.size())
            throw new IllegalArgumentException("ERROR: feats.ncols() != labels.ncols()");

        int nepochs = 1000;
        int nrows_feat = feats.get(0).nrows();
        int ncols_feat = feats.get(0).ncols();
        int nchannels_feat = feats.get(0).nchannels();

        Matkc labels = new Matkc(labels_);

        int[] idx_pos = labels.find("=", 1).indices;
        int[] idx_neg = labels.find("=", -1).indices;

        int npos = idx_pos.length;
        int nneg = idx_neg.length;

        Collections.shuffle(Ints.asList(idx_pos));
        Collections.shuffle(Ints.asList(idx_neg));

        int niters = Math.max(npos, nneg);

        List<Matkc> w_lin_all = new ArrayList<>(nepochs);
        List<Double> bias_all = new ArrayList<>(nepochs);
        List<Double> perf_all = new ArrayList<>(nepochs);

        // initialize linear weights and bias
        w_lin = new Matkc(nrows_feat, ncols_feat, nchannels_feat);
        bias = 0;

        int cc_pos = 0;
        int cc_neg = 0;
        boolean pick_pos = true; // to alternate positive and negative data points
        int idx_picked;

        double label_groundtruth;

        for (int i = 0; i < nepochs; i++)
        {
            int nwrongs = 0; // keep track of how many errors made in coming epoch
            System.out.println("Epoch = " + i);

            for (int j = 0; j < niters; j++)
            {
                // turn to pick a positive
                if (pick_pos)
                {
                    //System.out.println("Turn to pick pos sample");
                    // if have gone through all +ve examples, need to begin from the start
                    // but after random shuffling
                    if (cc_pos == npos)
                    {
                        //System.out.println("Entire pos set gone through. Restarting from beginning.");
                        cc_pos = 0;
                        Collections.shuffle(Ints.asList(idx_pos));
                    }
                    idx_picked = idx_pos[cc_pos];
                    cc_pos++;
                    pick_pos = false;
                }
                // turn to pick a positive
                else
                {
                    //System.out.println("Turn to pick neg sample");
                    // if have gone through all -ve examples, need to begin from the start
                    // but after random shuffling
                    if (cc_neg == nneg)
                    {
                        //System.out.printf("Entire neg set gone through. Restarting from beginning.");
                        cc_neg = 0;
                        Collections.shuffle(Ints.asList(idx_neg));
                    }
                    idx_picked = idx_neg[cc_neg];
                    cc_neg++;
                    pick_pos = true;
                }

                Matkc featVec = feats.get(idx_picked);
                label_groundtruth = labels.get(idx_picked);

                //System.out.println("Picking training data num " + idx_picked + " which has label " + label_groundtruth);

                // if wrong prediction, then update weight
                if (classify(featVec) * label_groundtruth <= 0)
                {
                    //System.out.println("Wrong classifier. Updating weights");
                    w_lin = w_lin.plus(featVec.mult(label_groundtruth));
                    bias += (label_groundtruth * 1.0);
                    nwrongs++;
                }

            } //end j (iter)

            w_lin_all.add(w_lin.copy_deep());
            bias_all.add(bias);
            perf_all.add((niters - nwrongs)*100.0 / niters); // accuracy
            System.out.println("Training accuracy after this epoch = " + perf_all.get(perf_all.size()-1) + "%");
            //if (nwrongs == 0)
            if (nwrongs <= 10)
            {
                System.out.println("Early stopping due to nwrongs threshold.");
                break; // Early stopping due to nwrongs threshold.
            }
        } // end i (epoch)

    }

    // save the classifier
    public void save(String fpath)
    {
        w_lin.save(fpath);
    }

    // load the classifier
    public void load(String fpath)
    {
        w_lin = Matkc.load(fpath, false);
    }

    public boolean is_loaded_or_trained()
    {
        return w_lin != null;
    }
}
