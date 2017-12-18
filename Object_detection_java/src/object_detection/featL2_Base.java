// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import KKH.StdLib.MatrixHeader;
import org.opencv.core.Mat;

abstract public class featL2_Base {
    abstract public float[] extract(float[] featChannelImage, MatrixHeader mH, boolean col_major);
    int get_ndimsFeat() { return ndims_feat; }
    protected int ndims_feat;
}
