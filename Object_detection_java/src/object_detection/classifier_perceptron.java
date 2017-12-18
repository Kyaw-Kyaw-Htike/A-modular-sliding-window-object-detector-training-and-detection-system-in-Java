// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import KKH.StdLib.stdfuncs;
import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class classifier_perceptron extends classifier_Base {

    private float[] w_lin;
    private float bias;

    /**
     * Classify a given feature vector
     * @param featVec should be a opencv Mat row vector of type CV_32FC1.
     * @return classification score
     */
    @Override
    public float classify(float[] featVec)
    {
        float score = bias;
        for(int i=0; i<featVec.length; i++)
            score += w_lin[i] * featVec[i];
        return score;
    }

    /**
     * Train a classifier
     * @param featSet
     * @param labels_
     */
    @Override
    public void train(List<float[]> featSet, List<Integer> labels_)
    {
        int nepochs = 1000;
        int ndims_feat = featSet.get(0).length;

        int[] labels_v = ArrayUtils.toPrimitive(labels_.toArray(new Integer[0]));

        int[] idx_pos = stdfuncs.find_indices(labels_v, x->x==1);
        int[] idx_neg = stdfuncs.find_indices(labels_v, x->x==-1);

        int npos = idx_pos.length;
        int nneg = idx_neg.length;

        Collections.shuffle(Ints.asList(idx_pos));
        Collections.shuffle(Ints.asList(idx_neg));

        int niters = Math.max(npos, nneg);

        List<float[]> w_lin_all = new ArrayList<float[]>(nepochs);
        List<Float> bias_all = new ArrayList<Float>(nepochs);
        List<Float> perf_all = new ArrayList<Float>(nepochs);

        // initialize linear weights and bias
        w_lin = new float[ndims_feat];
        bias = 0;

        int cc_pos = 0;
        int cc_neg = 0;
        boolean pick_pos = true; // to alternate positive and negative data points
        int idx_picked;

        float label_groundtruth;

        for (int i = 0; i < nepochs; i++)
        {
            int nwrongs = 0; // keep track of how many errors made in coming epoch
            System.out.println("Epoch = " + i);

            for (int j = 0; j < niters; j++)
            {
                //cout << "Epoch " << i << ", Iter " << j << endl;
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

                float[] featVec = featSet.get(idx_picked);
                label_groundtruth = (float)labels_v[idx_picked];

                //System.out.println("Picking training data num " + idx_picked + " which has label " + label_groundtruth);

                // if wrong prediction, then update weight
                if (classify(featVec) * label_groundtruth <= 0)
                {
                    //System.out.println("Wrong classifier. Updating weights");
                    w_lin = stdfuncs.plus(w_lin, stdfuncs.multiply(featVec, label_groundtruth));
                    bias += (label_groundtruth * 1.0);
                    nwrongs++;
                }

            } //end j (iter)

            w_lin_all.add(w_lin);
            bias_all.add(bias);
            perf_all.add((float)(niters - nwrongs)*100.0f / niters); // accuracy
            System.out.println("Training accuracy after this epoch = " + perf_all.get(perf_all.size()-1) + "%");
            //if (nwrongs == 0)
            if (nwrongs <= 10)
            {
                System.out.println("Early stopping due to nwrongs threshold.");
                break; // Early stopping due to nwrongs threshold.
            }
        } // end i (epoch)

    }
}
