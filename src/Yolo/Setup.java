package Yolo;

import Tools.GlobalVars;

import java.io.File;

public abstract class Setup {

    private static boolean isInit = false;

    private static final String networkCfgPath = "Res/network.cfg";
    private static final String namesCfgPath = "Res/names.cfg";
    private static final String weightsPath = "Res/weight.weights";

    public static void initYolo() {

        if(!isInit) {

            GlobalVars.isJar = Setup.class.getResourceAsStream("/" + networkCfgPath) != null;
            System.out.printf("Running from .JAR package: %b\n",GlobalVars.isJar);
            GlobalVars.loadAlphabet();
            GlobalVars.loadNetwork(networkCfgPath,weightsPath,namesCfgPath);

            File predictions = new File("Predictions/");

            if(!predictions.exists()) {
                //noinspection ResultOfMethodCallIgnored
                predictions.mkdir();
            }

            isInit = true;
        }
    }
}
