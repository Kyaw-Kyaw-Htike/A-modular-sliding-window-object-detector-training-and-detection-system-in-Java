// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import org.opencv.core.Mat;

import org.opencv.core.CvType;
import org.opencv.imgproc.Imgproc;

public class featL1_naive extends featL1_Base {
    public featL1_naive()
    {
        shrinkage = 1;
        featNChannels = 1;
    }

    @Override
    public float[] extract(Mat img, int[] ndims_featChannel)
    {
        Mat img_out = new Mat(img.size(), CvType.makeType(img.depth(), 1));
        Imgproc.cvtColor(img, img_out, Imgproc.COLOR_RGB2GRAY);
        img_out.convertTo(img_out, CvType.CV_32FC1);
        float[] feat = new float[img_out.rows() * img_out.cols() * img_out.channels()];
        img_out.get(0, 0, feat);
        ndims_featChannel[0] = img_out.rows();
        ndims_featChannel[1] = img_out.cols();
        return feat;
    }
}
