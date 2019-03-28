package Tools;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public abstract class Util {

    public static final Date DATE = new Date();

    public static void topK(float[] a, int n, int k, int[] index, int oA, int oI) {

        for(int i = 0; i < k; i++) {
            index[oI + i] = -1;
        }

        for(int i = 0; i < n; i++) {
            int curr = i;
            for(int j = 0; j < k; j++) {
                if(index[oI + j] < 0 || a[oA + curr] > a[oA + index[oI + j]]) {
                    int swap = curr;
                    curr = index[oI + j];
                    index[oI + j] = swap;
                }
            }
        }
    }

    public static String readFile(String fileName) {

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            StringBuilder sb = new StringBuilder();
            String s;

            while((s = reader.readLine()) != null) {
                sb.append(s);
            }

            return sb.toString().replace("\n","");
        }
        catch (Exception e) {
            System.out.println(String.format("Error trying to load file '%s'.",fileName));
            return null;
        }
    }

    public static int[] readMap(String fileName) {

        List<Integer> map = new ArrayList<>();

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            String s;

            while((s = reader.readLine()) != null) {
                int num = Integer.parseInt(s.strip());
                map.add(num);
            }

            int[] _map = new int[map.size()];

            for(int i = 0; i < map.size(); i++) {
                _map[i] = map.get(i);
            }
            return _map;
        }
        catch (Exception e) {
            System.out.println(String.format("Error trying to load file '%s'.",fileName));
            return null;
        }

    }

    public static boolean findArg(int argc, List<String> argv, String arg) {

        for(int i = 0; i < argc; i++) {

            String s = argv.get(i);

            if(s != null && s.equals(arg)) {
                argv.remove(i);
                return true;
            }
        }
        return false;
    }

    public static int findIntArg(int argc, List<String> argv, String arg, int def) {

        for(int i = 0; i < argc; i++) {

            String s = argv.get(i);

            if(s != null && s.equals(arg)) {

                int val = Integer.parseInt(argv.get(i+1));

                argv.remove(i);
                argv.remove(i);
                return val;
            }
        }
        return def;
    }

    public static float findFloatArg(int argc, List<String> argv, String arg, float def) {

        for(int i = 0; i < argc; i++) {

            String s = argv.get(i);

            if(s != null && s.equals(arg)) {

                float val = Float.parseFloat(argv.get(i+1).replace(",","."));

                argv.remove(i);
                argv.remove(i);
                return val;
            }
        }
        return def;
    }

    public static String findStringArg(int argc, List<String> argv, String arg, String def) {

        for(int i = 0; i < argc; i++) {

            String s = argv.get(i);

            if(s != null && s.equals(arg)) {

                String val = argv.get(i+1);

                argv.remove(i);
                argv.remove(i);
                return val;
            }
        }
        return def;
    }

    public static String baseCfg(String cfgFile) {

        String[] c = cfgFile.split("/");
        String[] name = c[c.length -1].split("\\.");

        StringBuilder sb = new StringBuilder();

        for(int i = 0; i < name.length - 1; i++) {
            sb.append(name[i]);
        }

        return sb.toString();
    }

    public static float sumArray(float[] a,int n, int oA) {

        float sum = 0;

        for(int i = 0; i < n; i++) {
            sum += a[oA + i];
        }
        return sum;
    }

    public static float meanArray(float[] a, int n, int oA) {

        return sumArray(a,n,oA)/n;
    }

    public static void meanArrays(float[][] a, int n, int els, float[] avg) {

        ArrayUtils.setArrayValue(avg,0);

        for(int j = 0; j < n; j++) {
            for(int i = 0; i < els; i++) {
                avg[i] += a[j][i];
            }
        }
        for(int i = 0; i < els; i++) {
            avg[i] /= n;
        }
    }

    public static float varianceArray(float[] a, int n, int oA) {

        int i;
        float sum = 0;

        float mean = meanArray(a, n,oA);

        for(i = 0; i < n; ++i) {
            sum += (a[i + oA] - mean) * (a[i + oA] - mean);
        }
        return sum/n;
    }

    public static float distArray(float[] a, float[] b, int n, int sub, int oA, int oB) {

        int i;
        float sum = 0;
        for(i = 0; i < n; i += sub) {

            sum += Math.pow(a[i + oA] - b[i + oB], 2);
        }
        return (float) Math.sqrt(sum);
    }

    public static float mseArray(float[] a, int n, int oA) {

        int i;
        float sum = 0;
        for(i = 0; i < n; ++i) {
            sum += a[i + oA] * a[i + oA];
        }
        return (float) Math.sqrt(sum/n);
    }

    public static void normalizeArray(float[] a, int n, int oA) {

        int i;
        float mu = meanArray(a,n,oA);
        float sigma = (float) Math.sqrt(varianceArray(a,n,oA));

        for(i = 0; i < n; ++i){
            a[i + oA] = (a[i + oA] - mu)/sigma;
        }
    }

    public static void translateArray(float[] a, int n, float s, int oA) {

        for(int i = 0;  i < n; i++) {
            a[i + oA] += s;
        }
    }

    public static float magArray(float[] a, int n, int oA) {

        float sum = 0;
        for(int i = 0; i < n; ++i){
            sum += a[i + oA]*a[i + oA];
        }
        return (float) Math.sqrt(sum);
    }

    public static void scaleArray(float[] a, int n, float s, int oA) {

        int i;
        for(i = 0; i < n; ++i){
            a[i + oA] *= s;
        }
    }

    public static int sampleArray(float[] a, int n, int oA) {

        float sum = sumArray(a, n,oA);
        scaleArray(a, n, 1.0f/sum,oA);


        float r = Rand.rand.nextFloat();
        int i;
        for(i = 0; i < n; ++i){
            r = r - a[i + oA];
            if (r <= 0) return i;
        }
        return n-1;
    }

    public static int maxIndex(int[] a, int n, int oA) {

        if(n <= 0) return -1;
        int i, max_i = 0;
        int max = a[oA];
        for(i = 1; i < n; ++i){
            if(a[i + oA] > max){
                max = a[i + oA];
                max_i = i;
            }
        }
        return max_i;
    }

    public static int maxIndex(float[] a, int n, int oA) {

        if(n <= 0) return -1;
        int i, max_i = 0;
        float max = a[oA];
        for(i = 1; i < n; ++i){
            if(a[i + oA] > max){
                max = a[i + oA];
                max_i = i;
            }
        }
        return max_i;
    }

    public static int intIndex(int[] a, int val, int n, int oA) {

        int i;
        for(i = 0; i < n; ++i){
            if(a[i + oA] == val) return i;
        }
        return -1;
    }

    public static float[][] oneHotEncode(float[] a, int n, int k, int oA) {

        float[][] t = new float[n][k];
        int index;

        for(int i = 0; i < n; i++) {

            index = (int) a[i + oA];
            t[i][index] = 1;
        }

        return t;
    }

    public static int constrain(int val, int min, int max) {

        if(val < min) {
            return min;
        }
        else if(val > max) {
            return max;
        }
        else {
            return val;
        }
    }

    public static float constrain(float val, float min, float max) {

        if(val < min) {
            return min;
        }
        else if(val > max) {
            return max;
        }
        else {
            return val;
        }
    }

    public static int countFields(String line) {

        return line.split(",").length;
    }

    public static float[] parseFields(String line, int n) {

        float[] field = new float[n];
        String[] separated = line.split(",");

        for(int i = 0; i < separated.length; i++) {
            try {
                field[i] = Float.parseFloat(separated[i].strip());
            }
            catch (Exception e) {
                field[i] = Float.NaN;
            }
        }
        return field;
    }




}
