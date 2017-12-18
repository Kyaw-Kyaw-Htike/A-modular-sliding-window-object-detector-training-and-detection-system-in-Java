// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.StdLib.Matk;
import KKH.StdLib.MatrixHeader;
import org.opencv.core.Mat;

abstract public class featL2_Base {
    // features corresponding two one sliding window
    abstract public Matk extract(Matk featChannelImage);
    int nrows_feat() { return nrows_feat; }
    int ncols_feat() { return ncols_feat; }
    int nchans_feat() { return nchans_feat; }
    protected int nrows_feat;
    protected int ncols_feat;
    protected int nchans_feat;
}
