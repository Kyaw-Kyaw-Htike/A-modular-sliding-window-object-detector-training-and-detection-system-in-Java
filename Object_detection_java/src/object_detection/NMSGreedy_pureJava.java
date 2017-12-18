// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import KKH.NMS.NMS_Greedy;
import com.google.common.primitives.Floats;
import org.apache.commons.lang3.ArrayUtils;
import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NMSGreedy_pureJava extends NMS_Base {

    private float overlap_thresh;

    public NMSGreedy_pureJava()
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

             


    }

}
