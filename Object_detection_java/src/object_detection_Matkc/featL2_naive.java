// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.StdLib.Matkc;

public class featL2_naive extends featL2_Base {

    public featL2_naive(int nrows_feat, int ncols_feat, int nchans_feat)
    {
        this.nrows_feat = nrows_feat;
        this.ncols_feat = ncols_feat;
        this.nchans_feat = nchans_feat;
    }

    @Override
    public Matkc extract(Matkc featChannelImage_roi)
    {
        return featChannelImage_roi;
    }

}

