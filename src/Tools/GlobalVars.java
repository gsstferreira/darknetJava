package Tools;

import Classes.Buffers.ByteBuffer;
import Classes.Data;
import Classes.Image;
import Classes.Network;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.util.List;

public abstract class GlobalVars {

    public static boolean isJar = false;

    private static Image[][] alphabet;
    private static Network network;
    private static List<String> names;

    private static ByteBuffer weightBytes;

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

    public static List<String> getNames() {
        return names;
    }

    public static void loadWeights(String weightFile) {

        try {

            BufferedInputStream stream;
            if(isJar) {
                stream = new BufferedInputStream(GlobalVars.class.getResourceAsStream("/" + weightFile));
            }
            else {
                stream = new BufferedInputStream(new FileInputStream(weightFile));
            }
            weightBytes = new ByteBuffer(stream.readAllBytes());
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
