package Yolo;

import Classes.*;
import Classes.Box;
import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Classes.Image;
import Yolo.Enums.ImType;
import Tools.GlobalVars;

import java.util.List;
import java.util.Objects;


public abstract class Detector {

    private static ImageDisplayer displayer;

    private static DetectionResult testDetector(String filename, float thresh, float hier_thresh) {

        long time1;
        long time2;

        float nms =.45f;

        Network net = GlobalVars.getNetwork();

        time1 = System.currentTimeMillis();
        System.out.print("Loading image...\t");
        Image im = Image.loadImageColor(filename,0,0);
        Image sized = im.letterbox(net.w,net.h);

        Layer l = net.layers[net.n - 1];

        FloatBuffer X = sized.data;
        time2 = System.currentTimeMillis();
        System.out.printf("done in %f seconds.\n",(time2 - time1)/1000.0f);

        time1 = System.currentTimeMillis();
        System.out.print("Predicting...\t\t");
        net.predict(X);
        time2 = System.currentTimeMillis();

        float procTime = (time2 - time1)/1000.0f;

        System.out.print(String.format("%s: Predicted in %f seconds.\n\n", filename,procTime));

        int nboxes = 0;
        IntBuffer b = new IntBuffer(1);
        b.put(0,nboxes);

        Detection[] dets = net.getBoxes( im.w, im.h, thresh, hier_thresh, null, 1, b);
        nboxes = b.get(0);

        Box.doNmsSort(dets,nboxes,l.classes,nms);

        List<Result> resultList = im.drawDetections(dets, nboxes, thresh, GlobalVars.getNames(), GlobalVars.getAlphabet(), l.classes);

        String[] newNamePath = filename.split("/");
        String oldName = newNamePath[newNamePath.length - 1];
        String newName = "prediction_" + oldName.replace(".jpg","");

        newName = filename.replace(oldName,newName);

        im.saveToDisk(Objects.requireNonNullElse(null,newName), ImType.JPG, 80);

        String finalNewName = newName + ".jpg";

        new Thread(() -> {
            if(displayer != null) {
                displayer.dispose();

            }
            displayer = new ImageDisplayer(finalNewName,im.w + 20,im.h + 20);
            displayer.display();
        }).start();

        return new DetectionResult(procTime,resultList,im.w,im.h);
    }

    public static DetectionResult runDetector(String imagePath, float thresh) {

        float hier_thresh = 0.5f;

        return testDetector(imagePath, thresh,hier_thresh);
    }
}
