// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import org.opencv.core.Mat;

public abstract class featL1_Base {
    abstract public float[] extract(Mat img, int[] ndims_featChannel);
    public int get_shrinkage() { return shrinkage; }
    public int get_nchannels() { return featNChannels; }
    protected int shrinkage; // must be 1, 4, 8, 12, etc
    protected int featNChannels;
}
