package Tools;

import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Classes.Arrays.LongArray;
import java.util.stream.IntStream;

public abstract class Buffers {

    public static void copy(FloatArray src, FloatArray dest, int size) {

        IntStream.range(0,size).parallel().forEach(i -> dest.set(i,src.get(i)));
    }

    public static void copy(IntArray src, IntArray dest, int size) {

        IntStream.range(0,size).parallel().forEach(i -> dest.set(i,src.get(i)));
    }

    public static FloatArray copy(FloatArray src, int size) {

        float[] arr = new float[size];

        IntStream.range(0,size).parallel().forEach(i -> {
            arr[i] = src.get(i);
        });
        return new FloatArray(arr);

//        for(int i = 0; i < size; i++) {
//            f.set(i,src.get(i));
//        }
    }

    public static IntArray copy(IntArray src, int size) {

        int[] arr = new int[size];

        IntStream.range(0,size).parallel().forEach(i -> {
            arr[i] = src.get(i);
        });
        return new IntArray(arr);

//        for(int i = 0; i < size; i++) {
//            f.set(i,src.get(i));
//        }
    }

    public static LongArray copy(LongArray src, int size) {

        long[] arr = new long[size];

        IntStream.range(0,size).parallel().forEach(i -> {
            arr[i] = src.get(i);
        });
        return new LongArray(arr);

//        for(int i = 0; i < size; i++) {
//            f.set(i,src.get(i));
//        }
    }

    public static FloatArray realloc(FloatArray src, int newSize) {

        FloatArray fb = new FloatArray(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.set(i,src.get(i));
        }
        return fb;
    }

    public static IntArray realloc(IntArray src, int newSize) {

        IntArray fb = new IntArray(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.set(i,src.get(i));
        }
        return fb;
    }

}
