package Tools;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public abstract class BufferUtil {

    public static void bufferCopy(FloatBuffer src, FloatBuffer dest, int size) {

        for(int i = 0; i < size; i++) {
            dest.put(i,src.get(i));
        }
    }

    public static void bufferCopy(IntBuffer src, IntBuffer dest, int size) {

        for(int i = 0; i < size; i++) {
            dest.put(i,src.get(i));
        }
    }

    public static void arrayCopy(String[] src, String[] dest, int size) {

        System.arraycopy(src,0,dest,0,size);
    }

    public static FloatBuffer offsetBuffer(FloatBuffer src, int offset) {

        FloatBuffer arr = FloatBuffer.wrap(src.array(),offset + src.arrayOffset(),src.array().length - offset - src.arrayOffset());

        return arr.slice();
    }

    public static IntBuffer offsetBuffer(IntBuffer src, int offset) {

        IntBuffer arr = IntBuffer.wrap(src.array(),offset + src.arrayOffset(),src.array().length - offset - src.arrayOffset());
        
        return arr.slice();
    }

    public static String[] offsetArray(String[] src, int offset) {

        String[] arr = new String[src.length - offset];

        System.arraycopy(src,offset,arr,0,src.length - offset);

        return arr;
    }
    
    public static void setBufferValue(IntBuffer array, int value) {

        int c = array.capacity();

        for(int i = 0; i < c; i++) {
            array.put(i,value);
        }
    }

    public static void setBufferValue(FloatBuffer array, float value) {

        int c = array.capacity();

        for(int i = 0; i < c; i++) {
            array.put(i,value);
        }
    }

    public static void setBufferValue(IntBuffer array, int value, int size) {

        for(int i = 0; i < size; i++) {
            array.put(i,value);
        }
    }

    public static void setBufferValue(FloatBuffer array, float value, int size) {

        for(int i = 0; i < size; i++) {
            array.put(i,value);
        }
    }


    public static FloatBuffer reallocBuffer(FloatBuffer src, int newSize) {

        FloatBuffer fb = org.lwjgl.BufferUtils.createFloatBuffer(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.put(i,src.get(i));
        }
        return fb;
    }

    public static IntBuffer reallocBuffer(IntBuffer src, int newSize) {

        IntBuffer fb = org.lwjgl.BufferUtils.createIntBuffer(newSize);

        for(int i = 0; i < newSize; i++) {
            fb.put(i,src.get(i));
        }
        return fb;
    }
}
