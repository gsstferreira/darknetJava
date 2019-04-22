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
            GlobalVars.loadAlphabet();
            GlobalVars.loadNetwork(networkCfgPath,weightsPath,namesCfgPath);

            new File("Predictions/").mkdir();

            isInit = true;
        }
    }
}
