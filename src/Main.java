
import Tools.Detector;

import java.io.*;
import java.util.List;
import java.util.stream.Collectors;


public class Main {

    public static void main(String[] args) {

        String cfgFile = "C:/darknetaws/cfg/yolov3-tiny.cfg";
        String weightFile = "C:/darknetaws/backup/yolov3-tiny_20000.weights";
        String dataFile = "C:/darknetaws/cfg/myYolo2.data";
        String imgPath = "C:/darknetaws/data/dual6.jpg";

        Detector.runDetector(dataFile,cfgFile,weightFile,imgPath);
    }
}
