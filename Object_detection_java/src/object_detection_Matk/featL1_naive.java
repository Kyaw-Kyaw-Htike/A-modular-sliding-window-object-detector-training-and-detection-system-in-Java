// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.StdLib.Matk;
import org.opencv.core.Mat;

import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;

// just raw pixels as features
public class featL1_naive extends featL1_Base {
    public featL1_naive()
    {
        shrinkage = 1;
        featNChannels = 1;
    }

    @Override
    public Matk extract(Matk img, int[] ndims_featChannel)
    {
        ndims_featChannel[0] = img.nrows();
        ndims_featChannel[1] = img.ncols();
        return img;
    }
}
