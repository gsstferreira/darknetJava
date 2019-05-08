package Yolo;

import Tools.Global;
import java.io.File;

public abstract class Setup {

    private static final String networkCfgPath = "Res/network.cfg";
    private static final String namesCfgPath = "Res/names.cfg";
    private static final String weightsPath = "Res/weight.bin";

    public static void initYolo() {

        Global.isJar = Setup.class.getResourceAsStream("/" + networkCfgPath) != null;

        if(!Global.isJar) {
            Global.saveResult = true;
            Global.displayResult = true;
        }


        System.out.printf("Running from .JAR package: %b\n", Global.isJar);
        System.out.printf("Saving result image file: %b\n", Global.saveResult);
        System.out.printf("Displaying result image: %b\n", Global.displayResult && Global.saveResult);
        Global.loadAlphabet();
        Global.loadNetwork(networkCfgPath,weightsPath,namesCfgPath);

        File predictions = new File("Predictions/");

        if(!predictions.exists()) {
            //noinspection ResultOfMethodCallIgnored
            predictions.mkdir();
        }
    }
}
