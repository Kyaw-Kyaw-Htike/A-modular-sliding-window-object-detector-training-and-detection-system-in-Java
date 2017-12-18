// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.StdLib.Matk;
import org.opencv.core.Rect;

import java.util.List;

abstract public class NMS_Base {

    protected Matk dr_nms;
    protected Matk ds_nms;

    abstract void suppress(Matk dr, Matk ds);
    public Matk get_dr_nms() { return dr_nms; }
    public Matk get_ds_nms() { return ds_nms; }
}
