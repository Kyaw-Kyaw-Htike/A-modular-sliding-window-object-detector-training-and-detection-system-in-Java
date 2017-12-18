// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.StdLib.Matk;
import org.opencv.core.Mat;

import java.util.List;

abstract public class classifier_Base {
    // feats is a list of Matk objects. Each Matk is a matrix.
    // can also be a vector.
    // labels is a list of labels that denote classes
    abstract void train(List<Matk> feats, List<Integer> labels);
    // featVec should be a matrix of size Mx1
    abstract double classify(Matk featVec);
    abstract void save(String fpath);
    abstract void load(String fpath);
    // check whether the classifier is loaded or trained
    abstract boolean is_loaded_or_trained();
}
