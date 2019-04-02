package Tools;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public abstract class BufferUtils {

    public static void bufferCopy(FloatBuffer src, FloatBuffer dest, int size) {

        System.arraycopy(src.array(),0,dest.array(),0,size);
    }

    public static void bufferCopy(IntBuffer src, IntBuffer dest, int size) {

        System.arraycopy(src.array(),0,dest.array(),0,size);
    }

    public static void arrayCopy(String[] src, String[] dest, int size) {

        System.arraycopy(src,0,dest,0,size);
    }

    public static FloatBuffer offsetBuffer(FloatBuffer src, int offset) {

        FloatBuffer arr = FloatBuffer.wrap(src.array(),offset,src.array().length - offset);

        return arr.slice();
    }

    public static IntBuffer offsetBuffer(IntBuffer src, int offset) {

        IntBuffer arr = IntBuffer.wrap(src.array(),offset,src.array().length - offset);
        
        return arr.slice();
    }

    public static String[] offsetArray(String[] src, int offset) {

        String[] arr = new String[src.length - offset];

        System.arraycopy(src,offset,arr,0,src.length - offset);

        return arr;
    }
    
    public static void setBufferValue(IntBuffer array, int value) {

        for(int i = 0; i < array.array().length; i++) {
            array.array()[i] = value;
        }
    }

    public static void setBufferValue(FloatBuffer array, float value) {

        for(int i = 0; i < array.array().length; i++) {
            array.array()[i] = value;
        }
    }
}
