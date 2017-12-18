// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import org.opencv.core.Rect;

import java.util.List;

public class NMS_naive extends NMS_Base {

    @Override
    public void suppress(List<int[]> dr, List<Float> ds)
    {
        dr_nms = dr;
        ds_nms = ds;
    }
}
