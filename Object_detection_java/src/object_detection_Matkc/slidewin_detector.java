// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.Opencv.cvUtilFuncs;
import KKH.StdLib.Matkc;
import KKH.StdLib.stdfuncs;
import KKH.TimerTT.TimerTT;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class slidewin_detector {
    private featL1_Base featL1_obj;
    private featL2_Base featL2_obj;
    private classifier_Base classifier_obj;
    private NMS_Base NMS_Obj;

    // ===================================
    // for original image: the following params can be set by the user
    // in the constructor
    // ===================================
    // size of detection window ([0]: num rows; [1]: num cols)
    private int[] winsize = new int[2];
    // stride; same size for both horizontal and vertical slides
    private int stride;
    // the ratio between two scales for sliding window.
    // the smaller, the finer the scales and thus, the more scales
    // needed to processed
    private double scaleratio;
    // in case user wants to limit the maximum number of scales
    // e.g. for very small objects in very large images.
    // normally, just set this number to a very large number (inf).
    private int max_nscales; // if want to limit number of scales

    // ===================================
    // for feat channel "image": automatically computed params
    // during the constructor
    // ===================================
    // how much shrinking does the L1 feature extraction perform
    private int shrinkage_channel;
    // the stride on the channel "image". This should map back to the
    // stride on the original image.
    private int stride_channel;
    // size of window on the channel "image". This should map back to the
    // winsize on the original image.
    private int[] winsize_channel = new int[2];
    // number of channels of the channel "image".
    private int nchannels_channel;
    private int nrows_featL2;
    private int ncols_featL2;
    private int nchans_featL2;

    // ===================================
    // output data of the private "process_img" method; these data correponds to
    // processing of a particular image size. These data members are written
    // after the "process_img" method is called. The process_img is a private method
    // that does multi-scale sliding window processing including feature extraction,
    // classification, etc. The user can call public methods which make use of this
    // "process_img" method to access important functionalities.
    // ===================================
    // store info about for what image size the following "parameters" has been
    // prepared for.
    private int nrows_img_prep, ncols_img_prep;
    // no. of scales that image sliding window must process
    private int num_scales;
    // vector of scales computed for sliding window; scales.size()==num_scales
    private double[] scales;
    // total no. of sliding windows for the image (across all the scales).
    private int nslidewins_total;
    // vector of sliding window rectangles. dr.size()==nslidewins_total
    private Matkc dr;
    // for each sliding window rectangle, which scale did it come from;
    // stores the index to std::vector<double>scales
    private List<Integer> idx2scale4dr;
    // vector of sliding window classification scores. ds.size()==nslidewins_total
    // ds will only be written if det_mode (which is one of the arguments to the
    // process_img method) is true.
    private Matkc ds;
    // matrix of features for randomly sampled sliding windows.
    // this will only be written if det_mode = false;
    private List<Matkc> feats_slidewin;

    // ===================================
    // Other data
    // ===================================
    // directory of cropped image patches for positive class for training classifier
    private String dir_pos;
    // directory of full negative images for training classifier
    private String dir_neg;
    // the file path to training data feature matrix and labels for training classifier
    private String traindata_fpath_save;

    // ===================================
    // Private member functions
    // ===================================

    private void check_params_constructor()
    {
        if (stride % shrinkage_channel != 0)
            throw new IllegalArgumentException("ERROR: stride MOD shrinkage_channel != 0.\n");
        if (winsize[0] % shrinkage_channel != 0)
            throw new IllegalArgumentException("ERROR: winsize[0] MOD shrinkage_channel != 0.\n");
        if (winsize[1] % shrinkage_channel != 0)
            throw new IllegalArgumentException("ERROR: winsize[1] MOD shrinkage_channel != 0.\n");
        if (nchannels_channel != nchans_featL2)
            throw new IllegalArgumentException("ERROR: nchannels_channel != nchans_featL2");
    }

    // given an image, it will process in a sliding window manner
    // and write the "outputs"/results to certain private data members.
    // In the method argument, if save_feats == true, feature matrix of
    // of the sliding window rectangles will be saved in the data member
    // "feats_slidewin", where each row is the feature vector for one
    // sliding window rectangle (after L1 + L2 feature extraction).
    // be careful however that for very large images and very high dimensional
    // features, memory requirements might be too large.
    // if apply_classifier is false, the classifier will not be applied for the
    // sliding window. This should only be used when for example only want
    // dr and feats_slidewin (when sampling features, dr, etc.)
    // dr will always be computed and saved in all cases.
    private void process_img(Mat img)
    {
        process_img(img, false, true);
    }

    private void process_img(Mat img, boolean save_feats, boolean apply_classifier)
    {
        nrows_img_prep = img.rows(); // to record the private data member
        ncols_img_prep = img.cols(); // to record the private data member
        int nrows_img = nrows_img_prep; // for use locally in this method
        int ncols_img = ncols_img_prep; // for use locally in this method

        // compute analytically how many scales there are for sliding window.
        // this formula gives the same answer as would be computed in a loop.
        num_scales = (int) Math.min(
            Math.floor(Math.log((double)nrows_img / winsize[0]) / Math.log(scaleratio)),
        Math.floor(Math.log((double)(ncols_img) / winsize[1]) / Math.log(scaleratio))) + 1;

        // preallocate for efficiency
        scales = new double[num_scales];

        // find a tight upper bound on total no. of sliding windows needed
        double stride_scale, nsw_rows, nsw_cols;
        int nslidewins_total_ub = 0;
        for (int s = 0; s < num_scales; s++)
        {
            stride_scale = stride*Math.pow(scaleratio, s);
            nsw_rows = Math.floor(nrows_img / stride_scale) - Math.floor(winsize[0] / stride) + 1;
            nsw_cols = Math.floor(ncols_img / stride_scale) - Math.floor(winsize[1] / stride) + 1;
            // Without the increment below, I get exact computation of number of sliding
            // windows, but just in case (to upper bound it)
            ++nsw_rows; ++nsw_cols;
            nslidewins_total_ub += (nsw_rows * nsw_cols);
        }

        // preallocate/reserve for speed
        dr = new Matkc(4, nslidewins_total_ub);
        idx2scale4dr = new ArrayList<Integer>(nslidewins_total_ub);
        if (apply_classifier) ds = new Matkc(1, nslidewins_total_ub);
        if (save_feats)
        {
            feats_slidewin = new ArrayList<Matkc>(nslidewins_total_ub);
        }

        // the resized image and the channel image
        Mat img_cur;
        Matkc H;
        Matkc feat_vec;
        int[] ndims_featChannel = new int[2];

        // reset counter for total number of sliding windows across all scales
        nslidewins_total = 0;

        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGR);
        img.convertTo(img, CvType.CV_32FC3);

        for (int s = 0; s < num_scales; s++)
        {
            // compute how much I need to scale the original image for this current scale s
            scales[s] = Math.pow(scaleratio, s);
            // get the resized version of the original image with the computed scale
            img_cur = new Mat();
            Imgproc.resize(img, img_cur, new Size(), 1.0 / scales[s], 1.0 / scales[s], Imgproc.INTER_LINEAR);

            // use L1 feature extractor to extract features from this resized image
            H = featL1_obj.extract(cvUtilFuncs.cvMat_to_Matkc(img_cur), ndims_featChannel);

            // run sliding window in the channel image space
            for (int j = 0; j < ndims_featChannel[1] - winsize_channel[1] + 1; j += stride_channel)
            {
                for (int i = 0; i < ndims_featChannel[0] - winsize_channel[0] + 1; i += stride_channel)
                {
                    // save the current sliding window rectangle after mapping back:
                    // (1) map from channel "image" space to image space (at this scale)
                    // (2) map back to image space at this scale to original scale

                    dr.set(Math.round(((j+1)*shrinkage_channel-shrinkage_channel)*scales[s]), 0, nslidewins_total);
                    dr.set(Math.round(((i+1)*shrinkage_channel - shrinkage_channel)*scales[s]), 1, nslidewins_total);
                    dr.set(Math.round((winsize[1]) * scales[s]), 2, nslidewins_total);
                    dr.set(Math.round((winsize[0]) * scales[s]), 3, nslidewins_total);

                    // stores which scale of the original image this dr comes from
                    idx2scale4dr.add(s);

                    // extract the roi patch (deep copy)
                    Matkc featChan_roi = H.get(i, i+winsize_channel[0]-1, j, j+winsize_channel[1]-1, 0, -1);

                    // Get the channel image patch according to this current sliding window
                    // rectangle, extract L2 features which will output a feature vector.
                    feat_vec = featL2_obj.extract(featChan_roi);

                    //cout << "feat_vec rows, cols and channels: " << feat_vec.rows << " " << feat_vec.cols << " " << feat_vec.channels() << endl;

                    // apply classifier on the feature vector and save it
                    //if (apply_classifier) ds.add(classifier_obj.classify(feat_vec));
                    if (apply_classifier) ds.set(classifier_obj.classify(feat_vec), nslidewins_total);

                    // save the extracted features
                    if (save_feats) feats_slidewin.add(feat_vec);

                    ++nslidewins_total;

                } // end j
            } //end i

        } //end s

        // get rid of extra columns (due to overestimation in nslidewins_total_ub)
        dr = dr.get_cols(0, nslidewins_total-1);
        if (apply_classifier) ds = ds.get_cols(0, nslidewins_total-1);

    } //end method

    public static class Result_detection
    {
        public Matkc dr;
        public Matkc ds;
    }

    // The constructor where the user need specify feature extraction,
    // classifier and NMS objects. Default params are set for sliding window scheme.
    // if user wants to change these default sliding window params,
    // use the method "set_params"
    public slidewin_detector(featL1_Base a, featL2_Base b, classifier_Base c, NMS_Base d)
    {
        featL1_obj = a;
        featL2_obj = b;
        classifier_obj = c;
        NMS_Obj = d;

        winsize[0] = 128; // num rows of detection window size
        winsize[1] = 64; // num cols of detection window size
        stride = 8;
        scaleratio = Math.pow(2, 1 / 8.0);
        max_nscales = Integer.MAX_VALUE;

        nrows_img_prep = 0;
        ncols_img_prep = 0;

        shrinkage_channel = featL1_obj.get_shrinkage();
        nchannels_channel = featL1_obj.get_nchannels();
        nrows_featL2 = featL2_obj.nrows_feat();
        ncols_featL2 = featL2_obj.ncols_feat();
        nchans_featL2 = featL2_obj.nchans_feat();
        stride_channel = stride / shrinkage_channel;
        winsize_channel[0] = winsize[0] / shrinkage_channel;
        winsize_channel[1] = winsize[1] / shrinkage_channel;

        check_params_constructor();
    }

    public slidewin_detector(featL1_Base a, featL2_Base b, classifier_Base c, NMS_Base d,
                             int winsize_nrows, int winsize_ncols)
    {
        this(a, b, c, d, winsize_nrows, winsize_ncols, 8, Math.pow(2, 1 / 8.0), Integer.MAX_VALUE);
    }

    public slidewin_detector(featL1_Base a, featL2_Base b, classifier_Base c, NMS_Base d,
                             int winsize_nrows, int winsize_ncols, int stride_)
    {
        this(a, b, c, d, winsize_nrows, winsize_ncols, stride_, Math.pow(2, 1 / 8.0), Integer.MAX_VALUE);
    }

    // The constructor where the user need specify feature extraction,
    // classifier and NMS objects, and also params for sliding window scheme.
    public slidewin_detector(featL1_Base a, featL2_Base b, classifier_Base c, NMS_Base d,
                             int winsize_nrows, int winsize_ncols, int stride_,
                             double scaleratio_, int max_nscales_)
    {
        featL1_obj = a;
        featL2_obj = b;
        classifier_obj = c;
        NMS_Obj = d;

        winsize[0] = winsize_nrows;
        winsize[1] = winsize_ncols;
        stride = stride_;
        scaleratio = scaleratio_;
        max_nscales = max_nscales_;

        nrows_img_prep = 0;
        ncols_img_prep = 0;

        shrinkage_channel = featL1_obj.get_shrinkage();
        nchannels_channel = featL1_obj.get_nchannels();
        nrows_featL2 = featL2_obj.nrows_feat();
        ncols_featL2 = featL2_obj.ncols_feat();
        nchans_featL2 = featL2_obj.nchans_feat();
        stride_channel = stride / shrinkage_channel;
        winsize_channel[0] = winsize[0] / shrinkage_channel;
        winsize_channel[1] = winsize[1] / shrinkage_channel;

        check_params_constructor();
    }

    public List<Matkc> get_feats_img(Mat img)
    {
        return get_feats_img(img, -1);
    }

    // get feature vectors from given image from multi-scale sliding window space.
    // Useful for initially sampling negatives for training detector, etc.
    // if nsamples=-1, no sampling; return all features
    // according to all sliding windows in order
    public List<Matkc> get_feats_img(Mat img, int nsamples)
    {
        // get all feats first
        process_img(img, true, false);

        if (nsamples < 0) return feats_slidewin;

        // prepare random number generator which will randomly
        // sample from the integer set {0,1,...,nslidewins_total-1}
        Random rand = new Random();
        int min_val = 0;
        int max_val = nslidewins_total - 1;
        int randomNumber;

        List<Matkc> feats_sampled = new ArrayList<>(nsamples);

        for (int i = 0; i < nsamples; i++)
        {
            randomNumber = rand.nextInt(max_val + 1 - min_val) + min_val;
            feats_sampled.add(feats_slidewin.get(randomNumber));
        }

        return feats_sampled;
    }

    public Matkc get_dr_img(Mat img)
    {
        return get_dr_img(img, -1);
    }

    // get rectangles from given image from multi-scale sliding window space.
    // if nsamples=-1, no sampling; return all sliding windows in order
    public Matkc get_dr_img(Mat img, int nsamples)
    {
        // to get all dr first
        process_img(img, false, false);

        // if no sampling, then just return everything
        if (nsamples < 0) return dr;

        // prepare random number generator which will randomly
        // sample from the integer set {0,1,...,nslidewins_total-1}
        Random rand = new Random();
        int min_val = 0;
        int max_val = nslidewins_total - 1;
        int randomNumber;

        //List<Matk> dr_sampled = new ArrayList<>(nsamples);
        Matkc dr_sampled = new Matkc(4, nsamples);
        for (int i = 0; i < nsamples; i++)
        {
            randomNumber = rand.nextInt(max_val + 1 - min_val) + min_val;
            dr_sampled.set_col(dr.get_col(randomNumber), i);
        }

        return dr_sampled;
    }

    public Result_detection detect(Mat img)
    {
        return detect(img, 0, true);
    }

    // detect objects on the given image with the classifier
    public Result_detection detect(Mat img, double dec_thresh, boolean apply_NMS)
    {
        if(!classifier_obj.is_loaded_or_trained())
            throw new IllegalArgumentException("ERROR: classifier is either not trained or loaded.");

        process_img(img, false, true);

        Result_detection res = new Result_detection();
        res.dr = dr;
        res.ds = ds;
        if (apply_NMS)
        {
            TimerTT timer_nms = new TimerTT();
            timer_nms.tic();
            NMS_Obj.suppress(res.dr, res.ds);
            System.out.println("Time taken for NMS = " + timer_nms.toc() + " secs");
            res.dr = NMS_Obj.get_dr_nms();
            res.ds = NMS_Obj.get_ds_nms();
        }

        return res;
    }

    // apply detector on a set of test images in a directory and save the results to
    // to specified directories
    public void detect(String dir_test, String dir_detsBboxes_output, String dir_detsVis_output, double dec_thresh, boolean apply_NMS)
    {
        if(dir_test == null)
            throw new IllegalArgumentException("ERROR: dir_test == null");

        // first delete any existing files in the given output directories
        stdfuncs.delete_files_in_dir(dir_detsBboxes_output);
        stdfuncs.delete_files_in_dir(dir_detsVis_output);

        stdfuncs.Result_fnames res_fnames = stdfuncs.dir_imgnames(dir_test);
        System.out.println("Number of test images = " + res_fnames.nfiles);

        for(int j=0; j<res_fnames.nfiles; j++)
        {
            System.out.format("Processing image [%d out of %d]: %s\n", j+1, res_fnames.nfiles, res_fnames.fnames[j]);
            Mat img = Imgcodecs.imread(res_fnames.fnames_fullpath[j]);
            TimerTT timer_det = new TimerTT();
            timer_det.tic();
            slidewin_detector.Result_detection res = detect(img, dec_thresh, apply_NMS);
            System.out.println("Time taken for detection including NMS = " + timer_det.toc() + " secs");
            List<int[]> bboxes = new ArrayList<>(res.dr.ncols());
            for (int i = 0; i < res.dr.ncols(); i++) {
                if (res.ds.get(i) > dec_thresh) {
                    int[] rect_cur = res.dr.get_col(i).vectorize_to_intArray();
                    if(dir_detsVis_output != null)
                    {
                        Point p1 = new Point(rect_cur[0], rect_cur[1]);
                        Point p2 = new Point(rect_cur[0] + rect_cur[2], rect_cur[1] + rect_cur[3]);
                        Imgproc.rectangle(img, p1, p2, new Scalar(255, 0, 0, 0), 2);
                    }
                    bboxes.add(rect_cur);
                }
            }
            if(dir_detsVis_output != null)
            {
                String fpath_detVis = dir_detsVis_output + res_fnames.fnames[j];
                Imgcodecs.imwrite(fpath_detVis, img);
                System.out.println("Detection visualization image saved at: " + fpath_detVis);
            }

            if(dir_detsBboxes_output != null)
            {
                String fpath_bboxData = dir_detsBboxes_output + res_fnames.fnames[j] + ".txt";
                try {
                    PrintWriter writer = new PrintWriter(fpath_bboxData, "UTF-8");
                    for(int i=0; i<bboxes.size(); i++)
                    {
                        int[] rect_cur = bboxes.get(i);
                        writer.println(rect_cur[0] + "," + rect_cur[1] + "," +
                                rect_cur[2] + "," + rect_cur[3]);
                    }
                    writer.close();
                }
                catch (IOException e) {
                    System.out.println(e.getStackTrace());
                }
                System.out.println("Detection bboxes text file saved at: " + fpath_bboxData);
            }
            System.out.println("========================");
        }
    }

    public static class Params_train
    {
        // if use_past_saved_feats = true, none of the options below
        // (apart from fpath_saved_feats) applies.
        public boolean use_past_saved_feats = false;
        // only applies if use_past_saved_feats = true
        public String fpath_saved_feats = "C:/Users/Kyaw/Desktop/detector_slidewin_savedFeats.bin";

        public int num_ini_negImg = 100;
        public int num_ini_nsamples_per_neg_img = 100;
        public boolean collect_hardnegs = true;
        // only applies if collect_hardnegs = true
        public int nhardnegs = 10000;
        public boolean save_collected_feats = false;
        // only applies if save_collected_feats = true
        public String fpath_save_feats = "C:/Users/Kyaw/Desktop/detector_slidewin_savedFeats.bin";
    }

    // train by extracting features from a directory of cropped positive patches, a directory of
    // full negative images (where hard negs will be mined). Optionally, all the extracted features
    // can be saved so that later on, if desired, I can use other overloaded train function
    // which just loads the saved features and labels for training
    public void train(String dir_pos_, String dir_neg_, Params_train params_train)
    {
        List<Matkc> feats_train;
        List<Integer> labels;

        if(!params_train.use_past_saved_feats)
        {
            System.out.println("Collecting data from scratch. Not using past saved features.");
            // just for recording so that in the future, I have a record of which training data
            // the detector was trained with
            dir_pos = dir_pos_;
            dir_neg = dir_neg_;

            // read in image full path names
            File fileObj;
            fileObj = new File(dir_pos_);
            String[] fnames_pos = fileObj.list((aa,bb)->{
                return bb.endsWith(".png") || bb.endsWith(".jpg") || bb.endsWith("jpeg") ||
                        bb.endsWith(".tiff") || bb.endsWith("tif");
            });
            fileObj = new File(dir_neg_);
            String[] fnames_neg = fileObj.list((aa,bb)->{
                return bb.endsWith(".png") || bb.endsWith(".jpg") || bb.endsWith("jpeg") ||
                        bb.endsWith(".tiff") || bb.endsWith("tif");
            });

            int npos = fnames_pos.length;
            int nnegImg = fnames_neg.length;

            // read cropped patches to form positive class of the dataset
            //Mat feats_pos = new Mat(npos, ndims_feat, CvType.CV_32FC1);
            List<Matkc> feats_pos = new ArrayList<>(npos);
            System.out.println("Extracting features from cropped +ve class...");
            for (int i = 0; i < npos; i++)
            {
                Mat img = Imgcodecs.imread(dir_pos + fnames_pos[i]);
                List<Matkc> ff = get_feats_img(img);
                if(ff.size() != 1)
                    throw new RuntimeException("ERROR: There are more than two patches here. Something's wrong.");
                feats_pos.add(ff.get(0));
            }
            System.out.println("Extracting +ve features done");
            System.out.println("feats_pos info: " + feats_pos.size() + " " + feats_pos.get(0).ndata());

            // random sample negative patches and features from negative images
            int num_ini_negImg = params_train.num_ini_negImg;
            int num_nsamples_per_img = params_train.num_ini_nsamples_per_neg_img;
            List<Matkc> feats_neg_ini = new ArrayList<>(num_ini_negImg);
            for (int i = 0; i < num_ini_negImg; i++)
            {
                Mat img = Imgcodecs.imread(dir_neg + fnames_neg[i]);
                List<Matkc> ff = get_feats_img(img, num_nsamples_per_img);
                feats_neg_ini.addAll(get_feats_img(img, num_nsamples_per_img));
            }
            System.out.println("feats_neg_ini info (# data points x feat dimension): " + feats_neg_ini.size() + " " +
                    feats_neg_ini.get(0).ndata());

            // train classifier with current initially collected dataset
            //Mat labels = new Mat(npos + feats_neg_ini.size(), 1, CvType.CV_32SC1);
            List<Integer> labels_pos = new ArrayList<>(Collections.nCopies(npos, 1));
            List<Integer> labels_neg = new ArrayList<>(Collections.nCopies(feats_neg_ini.size(), -1));
            labels = new ArrayList<>();
            labels.addAll(labels_pos);
            labels.addAll(labels_neg);

            feats_train = new ArrayList<>();
            feats_train.addAll(feats_pos);
            feats_train.addAll(feats_neg_ini);

            System.out.println("Training classifier...");
            classifier_obj.train(feats_train, labels);

            if(params_train.collect_hardnegs)
            {
                // go through negative images to find hard negs
                System.out.println("Collecting for hard negs...");
                int nhardnegs = params_train.nhardnegs;
                int nfp, tnfp;
                List<Matkc> feats_neg_hard = new ArrayList<>(nhardnegs);
                Mat img_roi;

                tnfp = 0;
                for (int i = 0; i < nnegImg; i++)
                {
                    Mat img = Imgcodecs.imread(dir_neg + fnames_neg[i]);
                    Result_detection res = detect(img, 0, true);
                    nfp = res.dr.ncols();

                    for (int j = 0; j < nfp; j++)
                    {
                        int[] r = res.dr.get_col(j).vectorize_to_intArray();
                        //cv::rectangle(img, dr_dets[j], cv::Scalar(255, 0, 0, 0), 2);
                        // just in case the given dr_deets[j] overshoots a bit (by 1 or 2 pixels)
                        // the image boundary, in order to prevent crashing
                        if (r[0] + r[2] >= img.cols() || r[1] + r[3] >= img.rows())
                            continue;
                        img_roi = new Mat(img, new Rect(r[0], r[1], r[2], r[3]));
                        feats_neg_hard.addAll(get_feats_img(img_roi, 1));
                        tnfp++;
                    }

                    System.out.println("Number of false +ves found in image " + i + " = " + nfp);
                    System.out.println("Total Number of false +ves collected so far = " + tnfp);

                    if (tnfp >= nhardnegs)
                    {
                        System.out.println("Stopped due to having reached target of " + nhardnegs + " hard neg samples");
                        break;
                    }
                }

                // add hardnegs to current dataset
                System.out.println("Adding hardnegs to the current dataset...");
                feats_train.addAll(feats_neg_hard);
                List<Integer> labels_hardneg = new ArrayList<>(Collections.nCopies(tnfp, -1));
                labels.addAll(labels_hardneg);
            }

            if(params_train.save_collected_feats)
            {
                System.out.println("Saving collected features at: " + params_train.fpath_save_feats);
                TimerTT timer_save = new TimerTT();
                timer_save.tic();
                stdfuncs.serialize_save(params_train.fpath_save_feats, new List[]{feats_train, labels});
                System.out.println("Collected features saved. Took " + timer_save.toc() + " secs.");
            }
        }

        else // use past saved features, therefore load these features
        {
            System.out.println("Loading past saved features from: " + params_train.fpath_saved_feats);
            TimerTT timer_load = new TimerTT();
            timer_load.tic();
            Object[] obj = (Object[])stdfuncs.serialize_load(params_train.fpath_saved_feats);
            feats_train = (List<Matkc>)obj[0];
            labels = (List<Integer>)obj[1];
            System.out.println("Past saved features loaded. Took " + timer_load.toc() + " secs.");
        }

        /*
        Given features to train, write a method that outputs the code to optimize for training
         a linear classifier. I can just paste this in the C++ code.
         Given the trained parameters output of the C++, load it in the java for object detection.

         HOG features

         go through each training data: applies logistic function with the weight vector
         then compute the loss function

         */

        // generate the C++ code for optimizating the logistic regression with the training dataset
        //

        //////////////////////////////////////////////
        ///////// C++ code generation begins ////////////////
        //////////////////////////////////////////////

        /*
        System.out.println("Generating C++ code for optmization..");
        System.out.format("Number of trainng data = %d, number of features = %d\n", feats_train.size(), feats_train.get(0).ndata());;

        // Charset.defaultCharset()
        // StandardOpenOption.WRITE
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get("C:/Users/Kyaw/Desktop/txt_out.txt"), StandardCharsets.UTF_8))
        {
            //writer.write(s, 0, s.length());
            writer.write("adouble calc_func_val(const adouble* x)\n{\n");
            writer.write("adouble  y = \n");
            double[] dataVec;
            for(int i=0; i<feats_train.size(); i++)
            {
                dataVec = feats_train.get(i).vectorize_to_doubleArray();
                int j;
                if(labels.get(i) == 1)
                    writer.write("adept::Log<adouble>((");
                else
                    writer.write("adept::Log<adouble>(1 - (");
                writer.write("1.0 / (1 + adept::Exp<adouble>(");
                for(j=0; j<feats_train.get(0).ndata()-1; j++)
                {
                    writer.write(String.format("x[%d]*%f + ", j, dataVec[j]));
                }
                writer.write(String.format("x[%d]*%f + x[%d]*1 )))) + \n", j, dataVec[j], j+1));
            }
            writer.write("0; \n");
            writer.write(String.format("return (-1/%d) * y; \n}", feats_train.size()));
        } catch (IOException ee) {
            System.err.format("ERROR: could not open file for writing IOException: %s%n", ee);
        }

        System.out.println("C++ code for optmization generated.");
        */

        //////////////////////////////////////////////
        ///////// C++ code generation ends ////////////////
        //////////////////////////////////////////////


        //////////////////////////////////////////////
        ///////// saving data for optimization begins ////////////////
        //////////////////////////////////////////////

        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get("C:/Users/Kyaw/Desktop/data_for_Cpp_optim.txt"), StandardCharsets.UTF_8))
        {
            double[] dataVec;
            for(int i=0; i<feats_train.size(); i++)
            {
                dataVec = feats_train.get(i).vectorize_to_doubleArray();
                writer.write(String.format("%d,", (int)labels.get(i)));
                int j;
                for(j=0; j<feats_train.get(0).ndata()-1; j++)
                {
                    writer.write(String.format("%f,", dataVec[j]));
                }
                writer.write(String.format("%f\n", dataVec[j]));
            }
        } catch (IOException ee) {
            System.err.format("ERROR: could not open file for writing IOException: %s%n", ee);
        }

        System.out.println("saving data for optimization completed.");



        //////////////////////////////////////////////
        ///////// saving data for optimization ends ////////////////
        //////////////////////////////////////////////

        System.out.println("Training classifier (final)...");
        TimerTT timer_train = new TimerTT();
        timer_train.tic();
        classifier_obj.train(feats_train, labels);
        System.out.println("Classifier training took " + timer_train.toc() + " secs.");
    }

    // save the trained detector (i.e. the trained classifier)
    public void save(String fpath)
    {
        classifier_obj.save(fpath);
    }

    // load the trained detector (i.e. the trained classifier)
    public void load(String fpath)
    {
        classifier_obj.load(fpath);
    }


}
