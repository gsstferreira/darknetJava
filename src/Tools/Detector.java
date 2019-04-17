package Tools;

import Classes.*;
import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Enums.ImType;

import java.util.List;


public abstract class Detector {

    public static DetectionResult testDetector(String namesFile, String cfgfile, String weightfile, String filename, float thresh, float hier_thresh, String outfile) {

        long time1;
        long time2;

        float nms =.45f;

        Network net = GlobalVars.getNetwork();

        time1 = System.currentTimeMillis();
        System.out.print("Loading image...\t\t\t\t");
        Image im = Image.loadImageColor(filename,0,0);
        Image sized = im.letterbox(net.w,net.h);

        Layer l = net.layers[net.n - 1];

        FloatBuffer X = sized.data;
        time2 = System.currentTimeMillis();
        System.out.printf("done in %f seconds.\n",(time2 - time1)/1000.0f);

        time1 = System.currentTimeMillis();
        System.out.print("Predicting...\t\t\t\t\t");
        net.predict(X);
        time2 = System.currentTimeMillis();


        float procTime = (time2 - time1)/1000.0f;

        System.out.print(String.format("%s: Predicted in %f seconds.\n", filename,procTime));

        int nboxes = 0;
        IntBuffer b = new IntBuffer(1);
        b.put(0,nboxes);

        Detection[] dets = net.getBoxes( im.w, im.h, thresh, hier_thresh, null, 1, b);
        nboxes = b.get(0);


        if (nms != 0) {

            Box.doNmsSort(dets,nboxes,l.classes,nms);
        }

        List<Result> resultList = im.drawDetections(dets, nboxes, thresh, GlobalVars.getNames(), GlobalVars.getAlphabet(), l.classes);

        if(outfile != null){
            im.saveToDisk(outfile, ImType.JPG,80);
        }
        else{
            im.saveToDisk("../predictions", ImType.JPG,80);
        }

        return new DetectionResult(procTime,resultList);
    }

    public static DetectionResult runDetector(String datacfg, String cfgfile, String weightfile, String filename) {

        float thresh = 0.5f;
        float hier_thresh = 0.5f;

        return testDetector(datacfg,cfgfile,weightfile,filename,thresh,hier_thresh,null);
    }
}
