package Tools;

public abstract class Setup {

    private static boolean isInit = false;

    private static final String networkCfgPath = "Res/network.cfg";
    private static final String namesCfgPath = "Res/names.cfg";
    private static final String weightsPath = "Res/weight.weights";

    public static void initYolo() {

        if(!isInit) {
            long time1 = System.currentTimeMillis();

            GlobalVars.isJar = Setup.class.getResourceAsStream("/" + networkCfgPath) != null;
            System.out.println(GlobalVars.isJar);
            GlobalVars.loadAlphabet();
            GlobalVars.loadNetwork(networkCfgPath,weightsPath,namesCfgPath);
            System.gc();

            long time2 = System.currentTimeMillis();

            isInit = true;
            System.out.printf("Setup time: %f seconds\n",(time2-time1)/1000.0f);
        }
    }
}
