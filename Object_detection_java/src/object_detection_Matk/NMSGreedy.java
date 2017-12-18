// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.NMS.NMS_Greedy;
import KKH.StdLib.Matk;
import com.google.common.primitives.Floats;
import org.apache.commons.lang3.ArrayUtils;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NMSGreedy extends NMS_Base {

    private float overlap_thresh;

    public NMSGreedy()
    {
        overlap_thresh = 0.5f;
    }

    public void set_thresh(float overlap_thresh)
    {
        this.overlap_thresh = overlap_thresh;
    }

    @Override
    public void suppress(Matk dr, Matk ds)
    {
        if(dr.ncols() != ds.length_vec())
            throw new IllegalArgumentException("dr.ncols() != ds.length_vec()");

        if(dr.ncols() == 0) return;

        int ndr = dr.ncols();

        float[] dr_ref = dr.t().vectorize_to_floatArray();
        float[] ds_ref = ds.vectorize_to_floatArray();

        NMS_Greedy nmsObj = new NMS_Greedy();
        nmsObj.suppress(dr_ref, ds_ref, overlap_thresh);

        float[] dr_nms_ref = nmsObj.get_dr_nms();
        float[] ds_nms_ref = nmsObj.get_ds_nms();
        int ndr_nms = ds_nms_ref.length;

        dr_nms = new Matk(dr_nms_ref, false, 4, ndr_nms, 1);
        ds_nms = new Matk(ds_nms_ref, false, 1, ndr_nms, 1);

    }

}
