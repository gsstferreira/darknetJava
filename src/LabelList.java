import Classes.Image;
import Yolo.Enums.ImType;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class LabelList {

    private static final String dataSetPath = "C:/Yolo/training/labels/";
    private static final String dataSetPathCw = "C:/Yolo/training/labels_cw/";
    private static final String dataSetPathCcw= "C:/Yolo/training/labels_ccw/";
    private static final String listOut = "C:/Yolo/training/";

    private static final int k_ratio = 10;

    public static void main(String[] args) {

        try {

            File f = new File(dataSetPath);
            File fCw = new File(dataSetPathCw);
            File fCcw = new File(dataSetPathCcw);

            String[] names = f.list((dir, name) -> name.contains(".jpg"));
            String[] namesCw = fCw.list((dir, name) -> name.contains(".jpg"));
            String[] namesCcw = fCcw.list((dir, name) -> name.contains(".jpg"));

            FileWriter writerTrain = new FileWriter(new File(listOut + "train_new.txt"));
            FileWriter writerTest = new FileWriter(new File(listOut + "test_new.txt"));

            int n = 0;

            if(names != null) {
                for(String s:names) {
                    String path = dataSetPath + s + "\n";
                    if(n%k_ratio == 0) {
                        writerTest.write(path);
                    }
                    else {
                        writerTrain.write(path);
                    }
                    n++;
                }
            }

            if(namesCw != null) {
                for(String s:namesCw) {
                    String path = dataSetPathCw + s + "\n";
                    if(n%k_ratio == 0) {
                        writerTest.write(path);
                    }
                    else {
                        writerTrain.write(path);
                    }
                    n++;
                }
            }

            if(namesCcw != null) {
                for(String s:namesCcw) {
                    String path = dataSetPathCcw + s + "\n";
                    if(n%k_ratio == 0) {
                        writerTest.write(path);
                    }
                    else {
                        writerTrain.write(path);
                    }
                    n++;
                }
            }

            writerTrain.close();
            writerTest.close();

        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
