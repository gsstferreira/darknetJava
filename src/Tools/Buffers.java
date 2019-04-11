package Tools;

import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Classes.Buffers.LongBuffer;

import java.nio.ByteBuffer;


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

    public static FloatBuffer copyNew(FloatBuffer src, int size) {

        FloatBuffer f = new FloatBuffer(size);

        for(int i = 0; i < size; i++) {
            f.put(i,src.get(i));
        }

        return f;
    }

    public static IntBuffer copyNew(IntBuffer src, int size) {

        IntBuffer f = new IntBuffer(size);

        for(int i = 0; i < size; i++) {
            f.put(i,src.get(i));
        }
        return f;
    }

    public static LongBuffer copyNew(LongBuffer src, int size) {

        LongBuffer f = new LongBuffer(size);

        for(int i = 0; i < size; i++) {
            f.put(i,src.get(i));
        }
        return f;
    }

    public static void copyArray(String[] src, String[] dest, int size) {

        System.arraycopy(src,0,dest,0,size);
    }

    public static String[] offset(String[] src, int offset) {

        String[] arr = new String[src.length - offset];

        System.arraycopy(src,offset,arr,0,src.length - offset);

        return arr;
    }
    
    public static void setValue(IntBuffer array, int value) {

        int c = array.size();

        for(int i = 0; i < c; i++) {
            array.put(i,value);
        }
    }

    public static void setValue(FloatBuffer array, float value) {

        int c = array.size();

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

        FloatBuffer fb = new FloatBuffer(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.put(i,src.get(i));
        }
        return fb;
    }

    public static IntBuffer realloc(IntBuffer src, int newSize) {

        IntBuffer fb = new IntBuffer(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.put(i,src.get(i));
        }
        return fb;
    }
    

    public static ByteBuffer reverse(ByteBuffer buffer) {

        int cap = buffer.limit() - buffer.position();
        ByteBuffer byteBuffer = ByteBuffer.allocate(cap);

        for(int i = 0; i < cap; i++) {

            byteBuffer.put(i,buffer.get(cap - i - 1));
        }
        return byteBuffer;
    }


}
