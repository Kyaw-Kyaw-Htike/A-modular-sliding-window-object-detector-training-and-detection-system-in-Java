// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import KKH.StdLib.MatrixHeader;
import KKH.StdLib.stdfuncs;
import org.opencv.core.Mat;

public class featL2_naive extends featL2_Base {

    public featL2_naive(int ndims_feat_)
    {
        ndims_feat = ndims_feat_;
    }

    @Override
    public float[] extract(float[] featChannelImage, MatrixHeader mH, boolean col_major)
    {
        if(col_major)
            return stdfuncs.extract_submat(featChannelImage, mH, false);
        else
            return stdfuncs.extract_submat(featChannelImage, mH, true);
    }

}

