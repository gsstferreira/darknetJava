package Tools;

import Classes.Data;
import Classes.DetectionResult;
import Classes.Image;
import Classes.Network;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.List;

public abstract class GlobalVars {

    private static Image[][] alphabet;
    private static Network network;
    private static List<String> names;

    private static byte[] weightBytes;
    private static int offset;

    public static BufferedWriter logStream;

    public static void loadAlphabet() {

        if(alphabet == null) {

            long time1 = System.currentTimeMillis();
            System.out.print("Loading labeling alphabet...\t");
            alphabet = Image.loadAlphabet();
            long time2 = System.currentTimeMillis();
            System.out.printf("done in %f seconds.\n",(time2 - time1)/1000.0f);
        }
        else {
            System.out.println("Alphabet already loaded!");
        }
    }

    public static void loadNetwork(String cfgFile, String weightFile, String namesFile) {

        if(network == null || names == null) {

            long time1 = System.currentTimeMillis();
            System.out.print("Loading network...\n");

            names = Data.getPaths(namesFile);

            network = Network.loadNetwork(cfgFile,weightFile,0);
            network.setBatchNetwork(1);
            long time2 = System.currentTimeMillis();
            System.out.printf("done in %f seconds.\n",(time2 - time1)/1000.0f);
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
            FileInputStream inputStream = new FileInputStream(new File(weightFile));
            BufferedInputStream stream = new BufferedInputStream(inputStream);

            weightBytes = stream.readAllBytes();
            stream.close();
            offset = 0;
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void freeWeights() {

        if(weightBytes != null) {
            weightBytes = null;
            offset = 0;
        }

        System.gc();
    }

    public static void setupLog(String logFile) {

        try {
            logStream = new BufferedWriter(new FileWriter(logFile));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static int getIntWeight() {

        return Buffers.reverse(ByteBuffer.wrap(getByteSection(4))).getInt(0);
    }

    public static float getFloatWeight() {

        return Buffers.reverse(ByteBuffer.wrap(getByteSection(4))).getFloat(0);
    }

    public static long getLongWeight() {
        return Buffers.reverse(ByteBuffer.wrap(getByteSection(8))).getLong(0);
    }

    private static byte[] getByteSection(int size) {

        byte[] b_arr = new byte[size];

        for(int i = 0; i < size; i++) {
            b_arr[i] = weightBytes[offset + i];
        }
        offset += size;
        return b_arr;
    }
}
