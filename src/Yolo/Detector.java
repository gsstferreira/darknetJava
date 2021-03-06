package Yolo;

import Classes.*;
import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Tools.Global;
import Yolo.Enums.ImType;

import java.util.List;
import java.util.Objects;

public abstract class Detector {

    private static final float nms = 0.45f;

    private static ImageDisplayer displayer;

    private static DetectionResult testDetector(String filename, float thresh, float hier_thresh) {

        long time1;
        long time2;

        Network net = Global.getNetwork();

        time1 = System.currentTimeMillis();
        System.out.print("Loading image...\t");
        Image im = Image.loadImageColor(filename,0,0,false);
        Image sized = im.letterbox(net.w,net.h);

        Layer l = net.layers[net.n - 1];

        FloatArray X = sized.data;
        time2 = System.currentTimeMillis();
        System.out.printf("done in %.3f seconds.\n",(time2 - time1)/1000.0f);

        time1 = System.currentTimeMillis();
        System.out.print("Predicting...\t\t");
        net.predict(X);
        time2 = System.currentTimeMillis();

        float procTime = (time2 - time1)/1000.0f;

        System.out.print(String.format("%s: Predicted in %.3f seconds.\n\n", filename,procTime));

        IntArray b = new IntArray(1);
        b.set(0,0);

        Detection[] dets = net.getBoxes( im.w, im.h, thresh, hier_thresh, null, 1, b);

        Box.doNmsSort(dets,b.get(0),l.classes,nms);

        List<Result> resultList = im.drawDetections(dets, b.get(0), thresh, Global.getNames(), Global.getAlphabet(), l.classes);

        if(Global.saveResult) {

            String[] newNamePath = filename.split("/");
            String oldName = newNamePath[newNamePath.length - 1];
            String newName = "Predictions/" + System.currentTimeMillis() + "_" + oldName.replace(".jpg", "");

            im.saveToDisk(Objects.requireNonNullElse(null,newName), ImType.JPG, 80);

            if(Global.displayResult) {

                String finalNewName = newName + ".jpg";
                new Thread(() -> {
                    if(displayer != null) {
                        displayer.updateImage(finalNewName,im.w + 20, im.h + 20);

                    }
                    else {
                        displayer = new ImageDisplayer(finalNewName,im.w + 20,im.h + 20);
                    }
                }).start();
            }
        }

        return new DetectionResult(procTime,thresh,resultList,im.w,im.h);
    }

    private static DetectionResult testDetectorImage(Image image, float thresh, float hier_thresh) {

        long time1;
        long time2;

        Network net = Global.getNetwork();

        time1 = System.currentTimeMillis();
        System.out.print("Loading image...\t");
        Image sized = image.letterbox(net.w,net.h);

        Layer l = net.layers[net.n - 1];

        FloatArray X = sized.data;
        time2 = System.currentTimeMillis();
        System.out.printf("done in %.3f seconds.\n",(time2 - time1)/1000.0f);

        time1 = System.currentTimeMillis();
        System.out.print("Predicting...\t\t");
        net.predict(X);
        time2 = System.currentTimeMillis();

        float procTime = (time2 - time1)/1000.0f;

        System.out.print(String.format("Predicted in %.3f seconds.\n\n", procTime));

        IntArray b = new IntArray(1);
        b.set(0,0);

        Detection[] dets = net.getBoxes( image.w, image.h, thresh, hier_thresh, null, 1, b);

        Box.doNmsSort(dets,b.get(0),l.classes,nms);

        List<Result> resultList = image.drawDetections(dets, b.get(0), thresh, Global.getNames(), Global.getAlphabet(), l.classes);

        if(Global.saveResult) {

            String newName = "Predictions/" + System.currentTimeMillis() + "_POSTreq";

            image.saveToDisk(Objects.requireNonNullElse(null,newName), ImType.JPG, 80);

            if(Global.displayResult) {

                String finalNewName = newName + ".jpg";
                new Thread(() -> {
                    if(displayer != null) {
                        displayer.updateImage(finalNewName,image.w + 20, image.h + 20);
                    }
                    else {
                        displayer = new ImageDisplayer(finalNewName,image.w + 20,image.h + 20);
                    }
                }).start();
            }
        }

        return new DetectionResult(procTime,thresh,resultList,image.w,image.h);
    }

    public static DetectionResult runDetector(String imagePath, float thresh) {

        float hierThresh = 0.5f;

        return testDetector(imagePath, thresh,hierThresh);
    }

    public static DetectionResult runDetectorImage(Image image, float thresh) {

        float hierThresh = 0.5f;

        return testDetectorImage(image, thresh,hierThresh);
    }
}
