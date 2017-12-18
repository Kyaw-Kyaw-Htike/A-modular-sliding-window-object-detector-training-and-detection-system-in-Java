// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import KKH.NMS.NMS_Greedy;
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
    public void suppress(List<int[]> dr, List<Float> ds)
    {
        if(dr.size() != ds.size())
            throw new IllegalArgumentException("dr.length != ds.length");

        if(dr.size() == 0) return;

        int ndr = dr.size();

        float[] dr_ref = new float[ndr * 4];
        float[] ds_ref = ArrayUtils.toPrimitive(ds.toArray(new Float[0]));

        int c = 0;
        for(int i=0; i<ndr; i++)
            dr_ref[c++] = (float)(dr.get(i)[0]);
        for(int i=0; i<ndr; i++)
            dr_ref[c++] = (float)(dr.get(i)[1]);
        for(int i=0; i<ndr; i++)
            dr_ref[c++] = (float)(dr.get(i)[2]);
        for(int i=0; i<ndr; i++)
            dr_ref[c++] = (float)(dr.get(i)[3]);

        NMS_Greedy nmsObj = new NMS_Greedy();
        nmsObj.suppress(dr_ref, ds_ref, overlap_thresh);

        float[] dr_nms_ref = nmsObj.get_dr_nms();
        float[] ds_nms_ref = nmsObj.get_ds_nms();
        int ndr_nms = ds_nms_ref.length;

        ds_nms = Arrays.asList(ArrayUtils.toObject(ds_nms_ref));

        dr_nms = new ArrayList<int[]>(ndr_nms);
        int x, y, w, h;
        for(int i=0; i<ndr_nms; i++)
        {
            int[] rect_cur = new int[4];
            rect_cur[0] = (int)dr_nms_ref[0 * ndr_nms + i];
            rect_cur[1] = (int)dr_nms_ref[1 * ndr_nms + i];
            rect_cur[2] = (int)dr_nms_ref[2 * ndr_nms + i];
            rect_cur[3] = (int)dr_nms_ref[3 * ndr_nms + i];
            dr_nms.add(rect_cur);
        }
    }

}
