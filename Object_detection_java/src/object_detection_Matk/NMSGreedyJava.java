// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.NMS.NMS_Greedy;
import KKH.StdLib.Matk;

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
    public void suppress(Matk dr, Matk ds)
    {
        if(dr.ncols() != ds.length_vec())
            throw new IllegalArgumentException("dr.ncols() != ds.length_vec()");

        dr_nms = new Matk(0, 0);
        ds_nms = new Matk(0, 0);

        if(dr.ncols() == 0) return;

        Matk x1 = dr.row(0).copy_deep();
        Matk y1 = dr.row(1).copy_deep();
        Matk x2 = x1.plus(dr.row(2));
        Matk y2 = y1.plus(dr.row(3));
        Matk s = ds.copy_deep();

        Matk area = x2.minus(x1).plus(1).multE(y2.minus(y1).plus(1));
        Matk.Result_sort res_sort = s.sort(false, true);
        Matk I = res_sort.indices_sort;
        Matk vals = res_sort.matSorted;

        Matk pick = new Matk(1, s.ncols());
        int counter = 0;

        Matk w, h, o, xx1, xx2, yy1, yy2, I_sub;

        while(I.ncols() > 0)
        {
            int last = I.ncols();
            int i = (int)I.get(last - 1);
            pick.set(i, counter);
            counter++;

            if (last == 1) break;
            I_sub = I.cols(0, last - 2);
            int[] I_sub_intArr = I_sub.to_int1DArray();

            xx1 = x1.cols(I_sub_intArr).max(x1.get(i));
            xx2 = x2.cols(I_sub_intArr).min(x2.get(i));
            yy1 = y1.cols(I_sub_intArr).max(y1.get(i));
            yy2 = y2.cols(I_sub_intArr).min(y2.get(i));

            w = xx2.minus(xx1).plus(1).max(0);
            h = yy2.minus(yy1).plus(1).max(0);

            o = w.multE(h).divE(area.cols(I_sub_intArr));
            Matk.Result_find res_find = o.find("<=", overlap_thresh);
            I = I.cols(res_find.indices);
        }

        pick = pick.cols(0, counter - 1);
        int[] picked_indices = pick.to_int1DArray();
        dr_nms = dr.cols(picked_indices);
        ds_nms = ds.cols(picked_indices);

    }

}
