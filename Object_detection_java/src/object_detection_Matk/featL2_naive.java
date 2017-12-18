// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.StdLib.Matk;
import KKH.StdLib.MatrixHeader;
import KKH.StdLib.stdfuncs;
import org.opencv.core.Mat;

public class featL2_naive extends featL2_Base {

    public featL2_naive(int nrows_feat, int ncols_feat, int nchans_feat)
    {
        this.nrows_feat = nrows_feat;
        this.ncols_feat = ncols_feat;
        this.nchans_feat = nchans_feat;
    }

    @Override
    public Matk extract(Matk featChannelImage_roi)
    {
        return featChannelImage_roi;
    }

}

