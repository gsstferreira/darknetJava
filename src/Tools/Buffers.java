package Tools;

import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public abstract class Buffers {

    public static void copy(FloatBuffer src, FloatBuffer dest, int size) {

        for(int i = 0; i < size; i++) {
            dest.put(i,src.get(i));
        }
    }

    public static void copy(IntBuffer src, IntBuffer dest, int size) {

        for(int i = 0; i < size; i++) {
            dest.put(i,src.get(i));
        }
    }

    public static void copyArray(String[] src, String[] dest, int size) {

        System.arraycopy(src,0,dest,0,size);
    }

    public static FloatBuffer offset(FloatBuffer src, int offset) {

        FloatBuffer arr = FloatBuffer.wrap(src.array(),offset + src.arrayOffset(),src.array().length - offset - src.arrayOffset());

        return arr.slice();
    }

    public static IntBuffer offset(IntBuffer src, int offset) {

        IntBuffer arr = IntBuffer.wrap(src.array(),offset + src.arrayOffset(),src.array().length - offset - src.arrayOffset());
        
        return arr.slice();
    }

    public static String[] offset(String[] src, int offset) {

        String[] arr = new String[src.length - offset];

        System.arraycopy(src,offset,arr,0,src.length - offset);

        return arr;
    }
    
    public static void setValue(IntBuffer array, int value) {

        int c = array.capacity();

        for(int i = 0; i < c; i++) {
            array.put(i,value);
        }
    }

    public static void setValue(FloatBuffer array, float value) {

        int c = array.capacity();

        for(int i = 0; i < c; i++) {
            array.put(i,value);
        }
    }

    public static void setValue(IntBuffer array, int value, int size) {

        for(int i = 0; i < size; i++) {
            array.put(i,value);
        }
    }

    public static void setValue(FloatBuffer array, float value, int size) {

        for(int i = 0; i < size; i++) {
            array.put(i,value);
        }
    }

    public static FloatBuffer realloc(FloatBuffer src, int newSize) {

        FloatBuffer fb = Buffers.newBufferF(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.put(i,src.get(i));
        }
        return fb;
    }

    public static IntBuffer realloc(IntBuffer src, int newSize) {

        IntBuffer fb = Buffers.newBufferI(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.put(i,src.get(i));
        }
        return fb;
    }

    public static FloatBuffer newBufferF(int capacity) {

        return BufferUtils.createFloatBuffer(capacity);
    }

    public static IntBuffer newBufferI(int capacity) {

        return BufferUtils.createIntBuffer(capacity);
    }
}
