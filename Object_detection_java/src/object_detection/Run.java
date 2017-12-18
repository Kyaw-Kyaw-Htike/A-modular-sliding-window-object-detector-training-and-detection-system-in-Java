// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.TrainData;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.Boost;

import KKH.Opencv.cvGUI;
import KKH.Opencv.cvUtilFuncs;
import KKH.TimerTT.TimerTT;
import KKH.StdLib.stdfuncs;
import KKH.StdLib.MatrixHeader;
import KKH.HogDollar.HogDollar;

import java.util.ArrayList;
import java.util.List;

public class Run {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {

        hogDollarFeatL1 featL1_obj = new hogDollarFeatL1(false, 8);
        featL2_naive featL2_obj = new featL2_naive(16*8*featL1_obj.get_nchannels());
        classifier_perceptron classifier_obj = new classifier_perceptron();
        NMSGreedy nms_obj = new NMSGreedy();

        slidewin_detector detector = new slidewin_detector(featL1_obj, featL2_obj, classifier_obj, nms_obj, 128, 64);

        String dir_pos = "D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context/";
        String dir_neg = "D:/Research/Datasets/INRIAPerson_Piotr/Train/images/set00/V001/";

        detector.train(dir_pos, dir_neg);

        // testing
        //Mat img = Imgcodecs.imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00001.png");
        Mat img = Imgcodecs.imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00000.png");
        List<int[]> dr = new ArrayList<>();
        List<Float> ds = new ArrayList<>();
        TimerTT timer_detect = new TimerTT();
        timer_detect.tic();
        detector.detect(img, dr, ds);
        System.out.println("Time taken = " + timer_detect.toc() + " secs");
        for (int i = 0; i < dr.size(); i++)
        {
        	if (ds.get(i) > 0)
            {
                int[] rect_cur = dr.get(i);
                Point p1 = new Point(rect_cur[0], rect_cur[1]);
                Point p2 = new Point(rect_cur[0]+rect_cur[2], rect_cur[1]+rect_cur[3]);
                Imgproc.rectangle(img, p1, p2, new Scalar(255, 0, 0, 0), 2);
            }
        }

        cvGUI.imshow(img);



    }
}
