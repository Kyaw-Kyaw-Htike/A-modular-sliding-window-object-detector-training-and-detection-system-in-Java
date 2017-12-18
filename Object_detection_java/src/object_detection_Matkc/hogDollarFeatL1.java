// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.HogDollar.HogDollar;
import KKH.StdLib.Matkc;

public class hogDollarFeatL1 extends featL1_Base {

    private HogDollar hogObj;

    public hogDollarFeatL1(boolean dalalHog, int shrinkage)
    {
        hogObj = new HogDollar();
        if (!dalalHog) hogObj.set_params_falzen_HOG();
        featNChannels = hogObj.nchannels_hog();
        this.shrinkage = shrinkage;
        hogObj.set_param_binSize(shrinkage);
    }

    @Override
    public Matkc extract(Matkc img, int[] ndims_featChannel)
    {
        int nr = img.nrows();
        int nc = img.ncols();
        int nch = img.nchannels();

        int nr_H = hogObj.nrows_hog(nr);
        int nc_H = hogObj.ncols_hog(nc);
        int nch_H = hogObj.nchannels_hog();

        float[] H = hogObj.extract(img.vectorize_to_floatArray(), nr, nc, nch);

        ndims_featChannel[0] = nr_H;
        ndims_featChannel[1] = nc_H;

        return new Matkc(H, true, nr_H, nc_H, nch_H);
    }
}
