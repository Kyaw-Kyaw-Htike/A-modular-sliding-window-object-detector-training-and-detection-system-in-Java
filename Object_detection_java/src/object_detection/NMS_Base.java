// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection;

import org.opencv.core.Rect;

import java.util.List;

abstract public class NMS_Base {

    protected List<int[]> dr_nms;
    protected List<Float> ds_nms;

    abstract void suppress(List<int[]> dr, List<Float> ds);
    public List<int[]> get_dr_nms() { return dr_nms; }
    public List<Float> get_ds_nms() { return ds_nms; }
}
