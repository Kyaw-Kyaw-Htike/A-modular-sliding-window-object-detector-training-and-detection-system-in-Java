// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import KKH.StdLib.MatrixHeader;
import KKH.StdLib.stdfuncs;
import KKH.TimerTT.TimerTT;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
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
    private int ndims_feat; // length of the feature vector after L2 feature extraction

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
    private List<int[]> dr;
    // for each sliding window rectangle, which scale did it come from;
    // stores the index to std::vector<double>scales
    private List<Integer> idx2scale4dr;
    // vector of sliding window classification scores. ds.size()==nslidewins_total
    // ds will only be written if det_mode (which is one of the arguments to the
    // process_img method) is true.
    private List<Float> ds;
    // matrix of features for randomly sampled sliding windows.
    // this will only be written if det_mode = false;
    private List<float[]> feats_slidewin;

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
        {
            throw new IllegalArgumentException("ERROR: stride MOD shrinkage_channel != 0.\n");
        }
        if (winsize[0] % shrinkage_channel != 0)
        {
            throw new IllegalArgumentException("ERROR: winsize[0] MOD shrinkage_channel != 0.\n");
        }
        if (winsize[1] % shrinkage_channel != 0)
        {
            throw new IllegalArgumentException("ERROR: winsize[1] MOD shrinkage_channel != 0.\n");
        }
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

        //cout << "nrows_img: " << nrows_img << " " << "ncols_img: " << ncols_img << endl;
        //cout << "num_scales: " << num_scales << endl;
        //cout << "nslidewins_total_ub: " << nslidewins_total_ub << endl;
        //cout << "num channels of L1 feat channel image: " << nchannels_channel << endl;
        //cout << "ndims_feat: " << ndims_feat << endl;

        // preallocate/reserve for speed
        dr = new ArrayList<int[]>(nslidewins_total_ub);
        idx2scale4dr = new ArrayList<Integer>(nslidewins_total_ub);
        if (apply_classifier) ds = new ArrayList<Float>(nslidewins_total_ub);
        if (save_feats)
        {
            feats_slidewin = new ArrayList<float[]>(nslidewins_total_ub);
        }

        // the resized image and the channel image
        Mat img_cur;
        float[] H;
        float[] feat_vec;
        int[] ndims_featChannel = new int[2];
        // reset counter for total number of sliding windows across all scales
        nslidewins_total = 0;

        for (int s = 0; s < num_scales; s++)
        {
            // compute how much I need to scale the original image for this current scale s
            scales[s] = Math.pow(scaleratio, s);
            // get the resized version of the original image with the computed scale
            img_cur = new Mat();
            Imgproc.resize(img, img_cur, new Size(), 1.0 / scales[s], 1.0 / scales[s], Imgproc.INTER_LINEAR);

            // use L1 feature extractor to extract features from this resized image
            H = featL1_obj.extract(img_cur, ndims_featChannel);

            // run sliding window in the channel image space
            for (int i = 0; i < ndims_featChannel[0] - winsize_channel[0] + 1; i += stride_channel)
            {
                for (int j = 0; j < ndims_featChannel[1] - winsize_channel[1] + 1; j += stride_channel)
                {
                    // save the current sliding window rectangle after mapping back:
                    // (1) map from channel "image" space to image space (at this scale)
                    // (2) map back to image space at this scale to original scale

                    int[] rect_cur = new int[4];
                    rect_cur[0] = (int)Math.round(((j+1)*shrinkage_channel-shrinkage_channel)*scales[s]);
                    rect_cur[1] = (int)Math.round(((i+1)*shrinkage_channel - shrinkage_channel)*scales[s]);
                    rect_cur[2] = (int)Math.round((winsize[1]) * scales[s]);
                    rect_cur[3] = (int)Math.round((winsize[0]) * scales[s]);

                    dr.add(rect_cur);

                    // stores which scale of the original image this dr comes from
                    idx2scale4dr.add(s);

                    MatrixHeader mH = new MatrixHeader(ndims_featChannel[0], ndims_featChannel[1], nchannels_channel);
                    mH.submat(i, i+winsize_channel[0]-1, j, j+winsize_channel[1]-1, 0, -1);

                    // Get the channel image patch according to this current sliding window
                    // rectangle, extract L2 features which will output a feature vector.
                    feat_vec = featL2_obj.extract(H, mH, true);

                    //cout << "feat_vec rows, cols and channels: " << feat_vec.rows << " " << feat_vec.cols << " " << feat_vec.channels() << endl;

                    // apply classifier on the feature vector and save it
                    if (apply_classifier) ds.add(classifier_obj.classify(feat_vec));

                    // save the extracted features
                    if (save_feats) feats_slidewin.add(feat_vec);

                    ++nslidewins_total;

                } // end j
            } //end i

        } //end s

        //cout << "nslidewins_total: " << nslidewins_total << endl;

    } //end method

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
        ndims_feat = featL2_obj.get_ndimsFeat();
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
                      double scaleratio_,int max_nscales_)
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
        ndims_feat = featL2_obj.get_ndimsFeat();
        stride_channel = stride / shrinkage_channel;
        winsize_channel[0] = winsize[0] / shrinkage_channel;
        winsize_channel[1] = winsize[1] / shrinkage_channel;

        check_params_constructor();
    }


    public List<float[]> get_feats_img(Mat img)
    {
        return get_feats_img(img, -1);
    }

    // get feature vectors from given image from multi-scale sliding window space.
    // Useful for initially sampling negatives for training detector, etc.
    // if nsamples=-1, no sampling; return all features
    // according to all sliding windows in order
    public List<float[]> get_feats_img(Mat img, int nsamples)
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

        List<float[]> feats_sampled = new ArrayList<>(nsamples);

        for (int i = 0; i < nsamples; i++)
        {
            randomNumber = rand.nextInt(max_val + 1 - min_val) + min_val;
            feats_sampled.add(feats_slidewin.get(randomNumber));
        }

        return feats_sampled;
    }


    public List<int[]> get_dr_img(Mat img)
    {
        return get_dr_img(img, -1);
    }

    // get rectangles from given image from multi-scale sliding window space.
    // if nsamples=-1, no sampling; return all sliding windows in order
    public List<int[]> get_dr_img(Mat img, int nsamples)
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

        List<int[]> dr_sampled = new ArrayList<int[]>(nsamples);
        for (int i = 0; i < nsamples; i++)
        {
            randomNumber = rand.nextInt(max_val + 1 - min_val) + min_val;
            dr_sampled.set(i, dr.get(randomNumber));
        }

        return dr_sampled;
    }

    public void detect(Mat img, List<int[]> dr_, List<Float> ds_)
    {
        detect(img, dr_, ds_, 0, true);
    }

    // detect objects on the given image with the classifier
    public void detect(Mat img, List<int[]> dr_, List<Float> ds_, float dec_thresh, boolean apply_NMS)
    {
        process_img(img, false, true);

        List<int[]> dr_temp;
        List<Float> ds_temp;

        if (apply_NMS)
        {
            List<int[]> dr_nms;
            List<Float> ds_nms;
            NMS_Obj.suppress(dr, ds);
            dr_temp = NMS_Obj.get_dr_nms();
            ds_temp = NMS_Obj.get_ds_nms();
        }
        else
        {
            dr_temp = dr;
            ds_temp = ds;
        }

        int nrects = dr_temp.size();
        dr_.clear();
        ds_.clear();

        for (int i = 0; i < nrects; i++)
        {
            if (ds_temp.get(i) > dec_thresh)
            {
                dr_.add(dr_temp.get(i));
                ds_.add(ds_temp.get(i));
            }
        }

    }

    // train by extracting features from a directory of cropped positive patches, a directory of
    // full negative images (where hard negs will be mined). Optionally, all the extracted features
    // can be saved so that later on, if desired, I can use other overloaded train function
    // which just loads the saved features and labels for training
    public void train(String dir_pos_, String dir_neg_)
    {
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
        List<float[]> feats_pos = new ArrayList<>(npos);
        System.out.println("Extracting features from cropped +ve class...");
        for (int i = 0; i < npos; i++)
        {
            Mat img = Imgcodecs.imread(dir_pos + fnames_pos[i]);
            List<float[]> ff = get_feats_img(img);
            if(ff.size() != 1)
                throw new RuntimeException("ERROR: There are more than two patches here. Something's wrong.");
            feats_pos.add(ff.get(0));
        }
        System.out.println("Extracting +ve features done");
        System.out.println("feats_pos info: " + feats_pos.size() + " " + feats_pos.get(0).length);

        // random sample negative patches and features from negative images
        int num_ini_negImg = 100;
        int num_nsamples_per_img = 100;
        List<float[]> feats_neg_ini = new ArrayList<>(num_ini_negImg);
        for (int i = 0; i < num_ini_negImg; i++)
        {
            Mat img = Imgcodecs.imread(dir_neg + fnames_neg[i]);
            List<float[]> ff = get_feats_img(img, num_nsamples_per_img);
            feats_neg_ini.addAll(get_feats_img(img, num_nsamples_per_img));
        }
        System.out.println("feats_neg_ini info: " + feats_neg_ini.size() + " " +
                feats_neg_ini.get(0).length);

        // train classifier with current initially collected dataset
        //Mat labels = new Mat(npos + feats_neg_ini.size(), 1, CvType.CV_32SC1);
        List<Integer> labels_pos = new ArrayList<>(Collections.nCopies(npos, 1));
        List<Integer> labels_neg = new ArrayList<>(Collections.nCopies(feats_neg_ini.size(), -1));
        List<Integer> labels = new ArrayList<>();
        labels.addAll(labels_pos);
        labels.addAll(labels_neg);

        List<float[]> feats_train = new ArrayList<>();
        feats_train.addAll(feats_pos);
        feats_train.addAll(feats_neg_ini);

        classifier_obj.train(feats_train, labels);

        // go through negative images to find hard negs
        System.out.println("Looking for hard negs...");
        int nhardnegs = 10000;
        //int nhardnegs = 1000;
        List<int[]> dr_dets = new ArrayList<>();
        List<Float> ds_dets = new ArrayList<>();
        int nfp, tnfp;
        List<float[]> feats_neg_hard = new ArrayList<>(nhardnegs);
        Mat img_roi;

        tnfp = 0;
        for (int i = 0; i < nnegImg; i++)
        {
            Mat img = Imgcodecs.imread(dir_neg + fnames_neg[i]);
            detect(img, dr_dets, ds_dets, 0, true);
            nfp = dr_dets.size();

            for (int j = 0; j < nfp; j++)
            {

                int[] r = dr_dets.get(j);
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

        // retrain classifier
        System.out.println("Retraining classifier with hard negs...");
        feats_train.addAll(feats_neg_hard);
        List<Integer> labels_hardneg = new ArrayList<>(Collections.nCopies(tnfp, -1));
        labels.addAll(labels_hardneg);

        System.out.println("Training classifier...");
        classifier_obj.train(feats_train, labels);
    }

}
