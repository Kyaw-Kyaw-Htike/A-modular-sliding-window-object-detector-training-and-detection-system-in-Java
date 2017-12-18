// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matkc;

import KKH.StdLib.Matkc;

public class NMSGreedyJava extends NMS_Base {

    private double overlap_thresh;

    public NMSGreedyJava()
    {
        overlap_thresh = 0.5;
    }

    public void set_thresh(float overlap_thresh)
    {
        this.overlap_thresh = overlap_thresh;
    }

    @Override
    public void suppress(Matkc dr, Matkc ds)
    {
        if(dr.ncols() != ds.length_vec())
            throw new IllegalArgumentException("dr.ncols() != ds.length_vec()");

        dr_nms = new Matkc(0, 0);
        ds_nms = new Matkc(0, 0);

        if(dr.ncols() == 0) return;

        Matkc x1 = dr.get_row(0);
        Matkc y1 = dr.get_row(1);
        Matkc x2 = x1.plus(dr.get_row(2));
        Matkc y2 = y1.plus(dr.get_row(3));
        Matkc s = ds.copy_deep();

        Matkc area = x2.minus(x1).plus(1).multE(y2.minus(y1).plus(1));
        Matkc.Result_sort res_sort = s.sort(false, true);
        Matkc I = res_sort.indices_sort;
        Matkc vals = res_sort.matSorted;

        Matkc pick = new Matkc(1, s.ncols());
        int counter = 0;

        Matkc w, h, o, xx1, xx2, yy1, yy2, I_sub;

        while(I.ncols() > 0)
        {
            int last = I.ncols();
            int i = (int)I.get(last - 1);
            pick.set(i, counter);
            counter++;

            if (last == 1) break;
            I_sub = I.get_cols(0, last - 2);
            int[] I_sub_intArr = I_sub.to_int1DArray();

            xx1 = x1.get_cols(I_sub_intArr).max(x1.get(i));
            xx2 = x2.get_cols(I_sub_intArr).min(x2.get(i));
            yy1 = y1.get_cols(I_sub_intArr).max(y1.get(i));
            yy2 = y2.get_cols(I_sub_intArr).min(y2.get(i));

            w = xx2.minus(xx1).plus(1).max(0);
            h = yy2.minus(yy1).plus(1).max(0);

            o = w.multE(h).divE(area.get_cols(I_sub_intArr));
            Matkc.Result_find res_find = o.find("<=", overlap_thresh);
            I = I.get_cols(res_find.indices);
        }

        pick = pick.get_cols(0, counter - 1);
        int[] picked_indices = pick.to_int1DArray();
        dr_nms = dr.get_cols(picked_indices);
        ds_nms = ds.get_cols(picked_indices);

    }

}
