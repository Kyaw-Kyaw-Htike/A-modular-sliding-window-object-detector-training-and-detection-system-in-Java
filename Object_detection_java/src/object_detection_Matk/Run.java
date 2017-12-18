// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import org.opencv.core.*;

public class Run {

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {

        String dir_pos = "D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context/";
        String dir_neg = "D:/Research/Datasets/INRIAPerson_Piotr/Train/images/set00/V001/";
        String fpath_save_detector = "C:/Users/Kyaw/Desktop/detector_slidewin.bin";
        boolean is_train_mode = true;

        hogDollarFeatL1 featL1_obj = new hogDollarFeatL1(false, 8);
        featL2_naive featL2_obj = new featL2_naive(16,8, featL1_obj.get_nchannels());
        //classifier_perceptron classifier_obj = new classifier_perceptron();
        //classifier_adaboost_jsat classifier_obj = new classifier_adaboost_jsat(1000);
        classifier_adaboost_weka classifier_obj = new classifier_adaboost_weka(1000);
        //NMSGreedy nms_obj = new NMSGreedy();
        NMSGreedyJava nms_obj = new NMSGreedyJava();

        slidewin_detector detector = new slidewin_detector(featL1_obj, featL2_obj, classifier_obj, nms_obj, 128, 64);

        slidewin_detector.Params_train params_train = new slidewin_detector.Params_train();
        //params_train.save_collected_feats = true;
        params_train.use_past_saved_feats = true;

        if(is_train_mode)
        {
            detector.train(dir_pos, dir_neg, params_train);
            detector.save(fpath_save_detector);
            System.out.println("Detector saved at: " + fpath_save_detector);
        }
        else
        {
            System.out.println("Loading detector from: " + fpath_save_detector);
            detector.load(fpath_save_detector);
        }

        String dir_test = "D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/";
        String dir_detsBboxes_output = "C:/Users/Kyaw/Desktop/detections/bboxes/";
        String dir_detsVis_output = "C:/Users/Kyaw/Desktop/detections/vis/";
        detector.detect(dir_test, dir_detsBboxes_output, dir_detsVis_output, 0, true);

        /*
        // testing on a single image
        //Mat img = Imgcodecs.imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00001.png");
        int N = 1;
        for(int k=0; k<N; k++) {
            Mat img = Imgcodecs.imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00000.png");
            TimerTT timer_det = new TimerTT();
            timer_det.tic();
            slidewin_detector.Result_detection res = detector.detect(img);
            System.out.println("Time taken for detection including NMS = " + timer_det.toc() + " secs");
            for (int i = 0; i < res.dr.ncols(); i++) {
                if (res.ds.get(i) > 0.55) {
                    int[] rect_cur = res.dr.col(i).vectorize_to_intArray();
                    Point p1 = new Point(rect_cur[0], rect_cur[1]);
                    Point p2 = new Point(rect_cur[0] + rect_cur[2], rect_cur[1] + rect_cur[3]);
                    Imgproc.rectangle(img, p1, p2, new Scalar(255, 0, 0, 0), 2);
                }
            }
            cvGUI.imshow(img);
        }
        */

        // *********************** Apply detector on images of test set *************************//
        /*
        String dir_test = "D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/";
        String dir_dets_output = "C:/Users/Kyaw/Desktop/detections/";
        stdfuncs.Result_fnames res_fnames = stdfuncs.dir_imgnames(dir_test);
        System.out.println("Number of test images = " + res_fnames.nfiles);
        for(int j=0; j<res_fnames.nfiles; j++)
        {
            System.out.println("Processing image: " + res_fnames.fnames[j]);
            Mat img = Imgcodecs.imread(res_fnames.fnames_fullpath[j]);
            TimerTT timer_det = new TimerTT();
            timer_det.tic();
            slidewin_detector.Result_detection res = detector.detect(img);
            System.out.println("Time taken for detection including NMS = " + timer_det.toc() + " secs");
            for (int i = 0; i < res.dr.ncols(); i++) {
                if (res.ds.get(i) > 0.55) {
                    int[] rect_cur = res.dr.col(i).vectorize_to_intArray();
                    Point p1 = new Point(rect_cur[0], rect_cur[1]);
                    Point p2 = new Point(rect_cur[0] + rect_cur[2], rect_cur[1] + rect_cur[3]);
                    Imgproc.rectangle(img, p1, p2, new Scalar(255, 0, 0, 0), 2);
                }
            }
//            cvGUI.imshow(img);
            Imgcodecs.imwrite(dir_dets_output + res_fnames.fnames[j], img);
        }
        */

    }
}
