package Tools;

import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public abstract class Util {

    public static final int SECRET_NUMBER = -1234;

//    public static final double log2 = Math.log(2);
//
//    public static void topK(FloatArray a, int n, int k, IntArray index) {
//
//        for(int i = 0; i < k; i++) {
//            index.set(i,-1);
//        }
//
//        for(int i = 0; i < n; i++) {
//            int curr = i;
//            for(int j = 0; j < k; j++) {
//                if(index.get(j) < 0 || a.get(curr) > a.get(index.get(j))) {
//                    int swap = curr;
//                    curr = index.get(j);
//                    index.set(j,swap);
//                }
//            }
//        }
//    }
//
//    public static String readFile(String fileName) {
//
//        try {
//
//            BufferedReader reader;
//            if(Global.isJar) {
//                reader = new BufferedReader(new InputStreamReader(Util.class.getResourceAsStream("/" + fileName)));
//            }
//            else {
//                reader = new BufferedReader(new FileReader(fileName));
//            }
//            StringBuilder sb = new StringBuilder();
//            String s;
//
//            while((s = reader.readLine()) != null) {
//                sb.append(s);
//            }
//            reader.close();
//
//            return sb.toString().replace("\n","");
//        }
//        catch (Exception e) {
//            System.out.println(String.format("Error trying to load file '%s'.",fileName));
//            return null;
//        }
//    }

    public static IntArray readMap(String fileName) {

        try {

            BufferedReader reader;
            if(Global.isJar) {
                reader = new BufferedReader(new InputStreamReader(Util.class.getResourceAsStream("/" + fileName)));
            }
            else {
                reader = new BufferedReader(new FileReader(fileName));
            }

            List<Integer> list = new ArrayList<>();
            String s;

            while((s = reader.readLine()) != null) {
                int num = Integer.parseInt(Util.strip(s));
                list.add(num);
            }
            reader.close();

            IntArray ib = new IntArray(list.size());

            for(int i = 0; i < ib.size(); i++) {
                ib.set(i,list.get(i));
            }

            return ib;
        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println(String.format("Error trying to load file '%s'.",fileName));
            return null;
        }
    }

//    public static boolean findArg(int argc, List<String> argv, String arg) {
//
//        for(int i = 0; i < argc; i++) {
//
//            String s = argv.get(i);
//
//            if(s != null && s.equals(arg)) {
//                argv.remove(i);
//                return true;
//            }
//        }
//        return false;
//    }
//
//    public static int findIntArg(int argc, List<String> argv, String arg, int def) {
//
//        for(int i = 0; i < argc; i++) {
//
//            String s = argv.get(i);
//
//            if(s != null && s.equals(arg)) {
//
//                int val = Integer.parseInt(argv.get(i+1));
//
//                argv.remove(i);
//                argv.remove(i);
//                return val;
//            }
//        }
//        return def;
//    }
//
//    public static float findFloatArg(int argc, List<String> argv, String arg, float def) {
//
//        for(int i = 0; i < argc; i++) {
//
//            String s = argv.get(i);
//
//            if(s != null && s.equals(arg)) {
//
//                float val = Float.parseFloat(argv.get(i+1).replace(",","."));
//
//                argv.remove(i);
//                argv.remove(i);
//                return val;
//            }
//        }
//        return def;
//    }
//
//    public static String findStringArg(int argc, List<String> argv, String arg, String def) {
//
//        for(int i = 0; i < argc; i++) {
//
//            String s = argv.get(i);
//
//            if(s != null && s.equals(arg)) {
//
//                String val = argv.get(i+1);
//
//                argv.remove(i);
//                argv.remove(i);
//                return val;
//            }
//        }
//        return def;
//    }
//
//    public static String baseCfg(String cfgFile) {
//
//        String[] c = cfgFile.split("/");
//        String[] name = c[c.length -1].split("\\.");
//
//        StringBuilder sb = new StringBuilder();
//
//        for(int i = 0; i < name.length - 1; i++) {
//            sb.append(name[i]);
//        }
//
//        return sb.toString();
//    }

    public static float sumArray(FloatArray a, int n) {

        float sum = 0;

        for(int i = 0; i < n; i++) {
            sum += a.get(i);
        }
        return sum;
    }

//    public static float meanArray(FloatArray a, int n) {
//
//        return sumArray(a,n)/n;
//    }

//    public static void meanArrays(float[][] a, int n, int els, float[] avg) {
//
//        FloatArray f = new FloatArray(avg);
//        Arrays.setAll(f,0);
//
//        for(int j = 0; j < n; j++) {
//            for(int i = 0; i < els; i++) {
//                avg[i] += a[j][i];
//            }
//        }
//        for(int i = 0; i < els; i++) {
//            avg[i] /= n;
//        }
//    }
//
//    public static float varianceArray(FloatArray a, int n) {
//
//        int i;
//        float sum = 0;
//
//        float mean = meanArray(a, n);
//
//        for(i = 0; i < n; ++i) {
//            sum += (a.get(i) - mean) * (a.get(i) - mean);
//        }
//        return sum/n;
//    }

//    public static float distArray(FloatArray a, FloatArray b, int n, int sub) {
//
//        int i;
//        float sum = 0;
//        for(i = 0; i < n; i += sub) {
//
//            sum += Math.pow(a.get(i) - b.get(i),2);
//        }
//        return (float) Math.sqrt(sum);
//    }
//
//    public static float mseArray(FloatArray a, int n) {
//
//        int i;
//        float sum = 0;
//        for(i = 0; i < n; ++i) {
//            sum += a.get(i) * a.get(i);
//        }
//        return (float) Math.sqrt(sum/n);
//    }
//
//    public static void normalizeArray(FloatArray a, int n) {
//
//        int i;
//        float mu = meanArray(a,n);
//        float sigma = (float) Math.sqrt(varianceArray(a,n));
//
//        for(i = 0; i < n; ++i){
//            a.set(i,(a.get(i) - mu)/sigma);
//        }
//    }
//
//    public static void translateArray(FloatArray a, int n, float s) {
//
//        for(int i = 0;  i < n; i++) {
//            a.set(i,a.get(i) +s);
//        }
//    }

    public static float magArray(FloatArray a, int n) {

        float sum = 0;
        for(int i = 0; i < n; ++i){

            sum += a.get(i) * a.get(i);
        }
        return (float) Math.sqrt(sum);
    }

//    public static void scaleArray(FloatArray a, int n, float s) {
//
//        int i;
//        for(i = 0; i < n; ++i){
//            a.set(i,a.get(i)*s);
//        }
//    }

//    public static int sampleArray(FloatArray a, int n) {
//
//        float sum = sumArray(a, n);
//        scaleArray(a, n, 1.0f/sum);
//
//
//        float r = Rand.randFloat();
//        int i;
//        for(i = 0; i < n; ++i){
//            r = r - a.get(i);
//            if (r <= 0) return i;
//        }
//        return n-1;
//    }
//
//    public static int maxIndex(IntArray a, int n) {
//
//        if(n <= 0) return -1;
//        int i, max_i = 0;
//        int max = a.get(0);
//
//        for(i = 1; i < n; ++i){
//            if(a.get(i) > max){
//                max = a.get(i);
//                max_i = i;
//            }
//        }
//        return max_i;
//    }
//
//    public static int maxIndex(FloatArray a, int n) {
//
//        if(n <= 0) return -1;
//
//        int i, max_i = 0;
//        float max = a.get(0);
//        for(i = 1; i < n; ++i){
//            if(a.get(i) > max){
//                max = a.get(i);
//                max_i = i;
//            }
//        }
//        return max_i;
//    }

    public static int intIndex(IntArray a, int val, int n) {

        int i;
        for(i = 0; i < n; ++i){
            if(a.get(i) == val) return i;
        }
        return -1;
    }

//    public static float[][] oneHotEncode(FloatArray a, int n, int k) {
//
//        float[][] t = new float[n][k];
//        int index;
//
//        for(int i = 0; i < n; i++) {
//
//            index = (int) a.get(i);
//            t[i][index] = 1;
//        }
//
//        return t;
//    }
//
//    public static int constrain(int val, int min, int max) {
//
//        if(val < min) {
//            return min;
//        }
//        else if(val > max) {
//            return max;
//        }
//        else {
//            return val;
//        }
//    }
//
//    public static float constrain(float val, float min, float max) {
//
//        if(val < min) {
//            return min;
//        }
//        else if(val > max) {
//            return max;
//        }
//        else {
//            return val;
//        }
//    }
//
//    public static int countFields(String line) {
//
//        return line.split(",").length;
//    }
//
//    public static FloatArray parseFields(String line, int n) {
//
//        float[] field = new float[n];
//        String[] separated = line.split(",");
//
//        for(int i = 0; i < separated.length; i++) {
//            try {
//                field[i] = Float.parseFloat(separated[i].strip());
//            }
//            catch (Exception e) {
//                field[i] = Float.NaN;
//            }
//        }
//        return new FloatArray(field);
//    }
//
//    public static byte[] toByteArray(int val) {
//
//        ByteArray byteBuffer = ByteArray.allocate(4);
//        byteBuffer.putInt(val);
//
//        return Arrays.reverse(byteBuffer).array();
//    }
//
//    public static byte[] toByteArray(float val) {
//
//        ByteArray byteBuffer = ByteArray.allocate(4);
//        byteBuffer.putFloat(val);
//
//        return Arrays.reverse(byteBuffer).array();
//    }
//
//    public static float byteToFloat(byte[] bytes) {
//
//        ByteArray byteBuffer = Arrays.reverse(ByteArray.wrap(bytes));
//        return byteBuffer.getFloat(0);
//    }
//
//    public static int byteToInt(byte[] bytes) {
//
//        ByteArray byteBuffer = Arrays.reverse(ByteArray.wrap(bytes));
//        return byteBuffer.getInt(0);
//    }

    public static String strip(String a) {

        String s = a;
        boolean start = true;
        int wSpace = 0;

        for(int i = 0; i < a.length(); i++) {

            if(Character.isWhitespace(a.charAt(i))) {
                wSpace++;
            }
            else {
                if(start) {
                    start = false;
                    s = a.substring(wSpace);
                }
                wSpace = 0;
            }

        }
        return s.substring(0,s.length() - wSpace);
    }
}
