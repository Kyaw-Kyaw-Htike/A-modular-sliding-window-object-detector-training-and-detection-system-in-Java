// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import KKH.HogDollar.HogDollar;
import KKH.StdLib.stdfuncs;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class hogDollarFeatL1 extends featL1_Base {

    private HogDollar hogObj;

    public hogDollarFeatL1(boolean dalalHog, int shrinkage_)
    {
        hogObj = new HogDollar();
        if (!dalalHog) hogObj.set_params_falzen_HOG();
        featNChannels = hogObj.nchannels_hog();
        shrinkage = shrinkage_;
        hogObj.set_param_binSize(shrinkage);
    }

    @Override
    public float[] extract(Mat img, int[] ndims_featChannel)
    {
        Mat img_temp = new Mat(img.size(), img.type());
        Imgproc.cvtColor(img, img_temp, Imgproc.COLOR_RGB2BGR);
        img_temp.convertTo(img_temp, CvType.CV_32FC3);

        int nr = img.rows();
        int nc = img.cols();
        int nch = img.channels();
        int nr_H = hogObj.nrows_hog(nr);
        int nc_H = hogObj.ncols_hog(nc);
        int nch_H = hogObj.nchannels_hog();

        float[] img_vec = new float[nr*nc*nch];
        float[] img_temp_data = new float[nr*nc*nch];
        img_temp.get(0, 0, img_temp_data);

        int cc = 0;
        for (int k = 0; k < nch; k++)
            for (int j = 0; j < nc; j++)
                for (int i = 0; i < nr; i++)
                    img_vec[cc++] = img_temp_data[stdfuncs.matPos_to_linearIndex_rowMajor(nr, nc, nch, i, j, k)];

        float[] H = hogObj.extract(img_vec, nr, nc, nch);

        ndims_featChannel[0] = nr_H;
        ndims_featChannel[1] = nc_H;

//        Mat feats_cv = new Mat(nr_H, nc_H, CvType.makeType(CvType.CV_32F, nch_H));
//        float[] feats_cv_data = new float[nr_H * nc_H * nch_H];
//
//        cc = 0;
//        for (int k = 0; k < nch_H; k++)
//            for (int j = 0; j < nc_H; j++)
//                for (int i = 0; i < nr_H; i++)
//                    feats_cv_data[StdUtils.mat_pos_to_RM_lin_index(nr_H, nc_H, nch_H, i, j, k)] = H[cc++];
//
//        feats_cv.put(0, 0, feats_cv_data);

//        return feats_cv;

        return H;
    }

}
