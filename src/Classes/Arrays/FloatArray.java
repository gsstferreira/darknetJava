package Classes.Arrays;

import java.util.stream.IntStream;

public class FloatArray {

    private final float[] array;
    private int offset;

    public FloatArray(float[] arr) {

        this.array = arr;
        this.offset = 0;
    }

    public FloatArray(int size) {

        this.array = new float[size];
        this.offset = 0;
    }

    public void offset(int off) {
        this.offset += off;

        if(this.offset < 0) {
            throw new IndexOutOfBoundsException("FloatArray offset is lesser than 0");
        }

        else if(this.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionArray offset is bigger than buffer length");
        }
    }

    public FloatArray offsetNew(int off) {

        FloatArray dec = new FloatArray(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("FloatArray offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionArray offset is bigger than buffer length");
        }

        return dec;
    }

    public float get(int index) {

        return this.array[index + offset];
    }

    public FloatArray shallowClone() {

        FloatArray buff = new FloatArray(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void put(int index,float d) {

        this.array[index + offset] = d;
    }

    public void setValue(float val) {

        IntStream.range(offset,array.length).forEach(i -> array[i] = val);
    }

    public void setValue(float val, int size) {

        IntStream.range(offset,offset + size).forEach(i -> array[i] = val);
    }

    public int size() {

        return this.array.length - this.offset;
    }

    public void copyInto(int size,int offsetSrc, FloatArray dest, int offsetdest) {

        IntStream.range(0,size).parallel().forEach(i -> dest.array[i*offsetdest] = this.array[i*offsetSrc]);
    }

    public void copyInto(int size, FloatArray dest) {

        IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i] = this.array[this.offset + i]);
    }
}
