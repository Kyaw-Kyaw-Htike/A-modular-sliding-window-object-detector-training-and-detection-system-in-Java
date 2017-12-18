// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import org.opencv.core.Mat;

import java.util.List;

abstract public class classifier_Base {
    abstract float classify(float[] featVec);
    abstract void train(List<float[]> featSet, List<Integer> labels);
    protected float thresh;
}
