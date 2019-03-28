package Tools;


public abstract class ArrayUtils {

    public static void arrayCopy(float[] src, float[] dest, int size) {

        System.arraycopy(src,0,dest,0,size);
    }

    public static void arrayCopy(int[] src, int[] dest, int size) {

        System.arraycopy(src,0,dest,0,size);
    }

    public static void arrayCopy(String[] src, String[] dest, int size) {

        System.arraycopy(src,0,dest,0,size);
    }

    public static float[] offsetArray(float[] src, int offset) {

        float[] arr = new float[src.length - offset];

        System.arraycopy(src,offset,arr,0,src.length - offset);

        return arr;
    }

    public static int[] offsetArray(int[] src, int offset) {

        int[] arr = new int[src.length - offset];

        System.arraycopy(src,offset,arr,0,src.length - offset);

        return arr;
    }

    public static String[] offsetArray(String[] src, int offset) {

        String[] arr = new String[src.length - offset];

        System.arraycopy(src,offset,arr,0,src.length - offset);

        return arr;
    }

    public static void undoOffsetArray(float[] originalArr, float[] offsetArr) {

        int offset = originalArr.length - offsetArr.length;

        System.arraycopy(offsetArr, 0, originalArr, offset, offsetArr.length);
    }

    public static void undoOffsetArray(int[] originalArr, int[] offsetArr) {

        int offset = originalArr.length - offsetArr.length;

        System.arraycopy(offsetArr, 0, originalArr, offset, offsetArr.length);
    }

    public static void undoOffsetArray(String[] originalArr, String[] offsetArr) {

        int offset = originalArr.length - offsetArr.length;

        System.arraycopy(offsetArr, 0, originalArr, offset, offsetArr.length);
    }

    public static void setArrayValue(int[] array, int value) {

        for(int i = 0; i < array.length; i++) {
            array[i] = value;
        }

    }

    public static void setArrayValue(float[] array, float value) {

        for(int i = 0; i < array.length; i++) {
            array[i] = value;
        }

    }

}
