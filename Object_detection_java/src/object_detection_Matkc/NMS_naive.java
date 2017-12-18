// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.StdLib.Matkc;

public class NMS_naive extends NMS_Base {

    @Override
    public void suppress(Matkc dr, Matkc ds)
    {
        dr_nms = dr;
        ds_nms = ds;
    }
}
