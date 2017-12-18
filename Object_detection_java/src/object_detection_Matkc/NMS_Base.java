// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.StdLib.Matkc;

abstract public class NMS_Base {

    protected Matkc dr_nms;
    protected Matkc ds_nms;

    abstract void suppress(Matkc dr, Matkc ds);
    public Matkc get_dr_nms() { return dr_nms; }
    public Matkc get_ds_nms() { return ds_nms; }
}
