package Yolo;

import Tools.GlobalVars;

import java.io.File;

public abstract class Setup {

    public  static boolean initDone = false;

    private static final String networkCfgPath = "Res/network.cfg";
    private static final String namesCfgPath = "Res/names.cfg";
    private static final String weightsPath = "Res/weight.weights";

    public static void initYolo() {

        if(!initDone) {

            GlobalVars.isJar = Setup.class.getResourceAsStream("/" + networkCfgPath) != null;
            System.out.printf("Running from .JAR package: %b\n",GlobalVars.isJar);
            System.out.printf("Displaying result image on detection run: %b\n",GlobalVars.showResultImage);
            GlobalVars.loadAlphabet();
            GlobalVars.loadNetwork(networkCfgPath,weightsPath,namesCfgPath);

            File predictions = new File("Predictions/");

            if(!predictions.exists()) {
                //noinspection ResultOfMethodCallIgnored
                predictions.mkdir();
            }

            initDone = true;
        }
    }
}
