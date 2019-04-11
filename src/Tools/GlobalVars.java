package Tools;

import Classes.DetectionResult;
import Classes.Image;

import java.io.*;
import java.nio.ByteBuffer;

public abstract class GlobalVars {

    public static Image[][] alphabet;
    public static float predictionTime;

    private static byte[] weightBytes;
    private static int offset;

    public static BufferedWriter logStream;

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
