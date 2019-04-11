package Classes.Buffers;

public class FloatBuffer {

    private float[] array;
    private int offset;

    public FloatBuffer(float[] arr) {

        this.array = arr;
        this.offset = 0;
    }

    public FloatBuffer(int size) {

        this.array = new float[size];
        this.offset = 0;
    }

    public void offset(int off) {
        this.offset += off;

        if(this.offset < 0) {
            throw new IndexOutOfBoundsException("FloatBuffer offset is lesser than 0");
        }

        else if(this.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is bigger than buffer length");
        }
    }

    public FloatBuffer offsetNew(int off) {

        var dec = new FloatBuffer(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("FloatBuffer offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is bigger than buffer length");
        }

        return dec;
    }

    public float get(int index) {

        return this.array[index + offset];
    }

    public void put(int index,float d) {

        this.array[index + offset] = d;
    }

    public int size() {

        return this.array.length - this.offset;
    }
}
