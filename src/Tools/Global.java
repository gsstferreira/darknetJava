package Tools;

import Classes.Arrays.ByteArray;
import Classes.Data;
import Classes.Image;
import Classes.Network;

import java.io.BufferedInputStream;
import java.io.FileInputStream;

public abstract class Global {

    public static boolean isJar = false;
    public static boolean saveResult = false;
    public static boolean displayResult = false;

    private static Image[][] alphabet;
    private static Network network;
    private static String[] names;

    private static ByteArray weightBytes;

    public static void loadAlphabet() {

        if(alphabet == null) {

            long time1 = System.currentTimeMillis();
            System.out.print("Loading labeling alphabet...\t");
            alphabet = Image.loadAlphabet();
            long time2 = System.currentTimeMillis();
            System.out.printf("done in %.3f seconds.\n",(time2 - time1)/1000.0f);
        }
        else {
            System.out.println("Alphabet already loaded!");
        }
    }

    public static void loadNetwork(String cfgFile, String weightFile, String namesFile) {

        if(network == null || names == null) {

            long time1 = System.currentTimeMillis();
            System.out.print("Loading network...\n\n");

            names = Data.getPaths(namesFile);

            network = Network.loadNetwork(cfgFile,weightFile,0);
            network.setBatchNetwork(1);
            long time2 = System.currentTimeMillis();
            System.out.printf("\nDone in %.3f seconds.\n",(time2 - time1)/1000.0f);
            Rand.setRandSeed(2222222);
        }
        else {
            System.out.println("Network already loaded!");
        }
    }

    public static Image[][] getAlphabet(){
        return alphabet;
    }

    public static Network getNetwork() {
        return network;
    }

    public static String[] getNames() {
        return names;
    }

    public static void loadWeights(String weightFile) {

        try {

            BufferedInputStream stream;
            if(isJar) {
                stream = new BufferedInputStream(Global.class.getResourceAsStream("/" + weightFile));
            }
            else {
                stream = new BufferedInputStream(new FileInputStream(weightFile));
            }
            weightBytes = new ByteArray(stream.readAllBytes());
            stream.close();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void freeWeights() {

        if(weightBytes != null) {
            weightBytes = null;
        }

        System.gc();
    }

    public static int getIntWeight() {

        int val = weightBytes.getNextInt(0);
        weightBytes.offset(4);
        return val;
    }

    public static float getFloatWeight() {

        float val = weightBytes.getNextFloat(0);
        weightBytes.offset(4);
        return val;
    }

    public static long getLongWeight() {

        long val = weightBytes.getNextLong(0);
        weightBytes.offset(8);
        return val;
    }
}
