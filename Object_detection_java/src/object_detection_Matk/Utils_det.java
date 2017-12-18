// Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

package object_detection_Matk;

import KKH.StdLib.Matk;
import jsat.classifiers.CategoricalData;
import jsat.classifiers.ClassificationDataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class Utils_det {
    public static ClassificationDataSet build_jsat_binary_classification_dataset(List<Matk> feats, List<Integer> labels, boolean adjust_class_imbalance)
    {
        int ndata = feats.size();
        List<DataPoint> data_jsat = new ArrayList<>(ndata);

        CategoricalData[] cat_data = new CategoricalData[1];
        cat_data[0] = new CategoricalData(2);
        cat_data[0].setCategoryName("Output");
        cat_data[0].setOptionName("Object", 0);
        cat_data[0].setOptionName("Non-object", 1);

        Matk temp = new Matk(labels);
        Matk.Result_find res_find_pos = temp.find("=", 1);
        Matk.Result_find res_find_neg = temp.find("=", -1);
        double weight_pos;
        if(adjust_class_imbalance)
            weight_pos = (double)res_find_neg.nFound / res_find_pos.nFound;
        else
            weight_pos = 1;
        System.out.println("Weight pos (to counter class imbalance) = " + weight_pos);

        System.out.println("Building dataset for JSAT");
        boolean is_pos;
        for(int i=0; i<ndata; i++)
        {
            Vec v = new DenseVector(feats.get(i).vectorize_to_doubleArray());
            int[] cat_val = new int[1];
            is_pos = labels.get(i) == 1;
            cat_val[0] = is_pos ? 0 : 1;
            data_jsat.add(new DataPoint(v, cat_val, cat_data, is_pos ? weight_pos : 1));
        }

        return new ClassificationDataSet(data_jsat, 0);
    }


    public static Instances build_weka_binary_classification_dataset(List<Matk> feats, List<Integer> labels, boolean adjust_class_imbalance)
    {
        int ndata = feats.size();
        int ndims_feat = feats.get(0).ndata();
        ArrayList<Attribute> attInfo = new ArrayList<>(ndims_feat+1);
        for(int i=0; i<ndims_feat; i++)
            attInfo.add(new Attribute("Feature " + (i+1)));

        List<String> class_names = new ArrayList<>(2);
        class_names.add("Object");
        class_names.add("Non-object");
        attInfo.add(new Attribute("Class", class_names));

        Matk temp = new Matk(labels);
        Matk.Result_find res_find_pos = temp.find("=", 1);
        Matk.Result_find res_find_neg = temp.find("=", -1);
        double weight_pos;
        if(adjust_class_imbalance)
            weight_pos = (double)res_find_neg.nFound / res_find_pos.nFound;
        else
            weight_pos = 1;
        System.out.println("Weight pos (to counter class imbalance) = " + weight_pos);

        Instances dataset_weka = new Instances("Dataset", attInfo, ndata);
        dataset_weka.setClassIndex(ndims_feat);
        boolean is_pos;
        double[] v = new double[ndims_feat+1];
        for(int i=0; i<ndata; i++)
        {
            is_pos = labels.get(i) == 1;
            System.arraycopy(feats.get(i).vectorize_to_doubleArray(), 0, v, 0, ndims_feat);
            v[ndims_feat] = is_pos ? 0 : 1;
            dataset_weka.add(new DenseInstance(is_pos ? weight_pos : 1, v));
        }

        return dataset_weka;
    }
}
