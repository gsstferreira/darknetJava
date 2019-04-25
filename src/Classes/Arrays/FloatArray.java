package Classes.Arrays;

import java.util.stream.IntStream;

public class FloatArray extends ArrayBase {

    private final float[] array;

    public FloatArray(float[] arr) {

        this.array = arr;
        this.size = arr.length;
        this.offset = 0;
    }

    public FloatArray(int size) {

        this.array = new float[size];
        this.size = size;
        this.offset = 0;
    }

    public FloatArray offsetNew(int off) {

        FloatArray dec = new FloatArray(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("FloatArray offset is lesser than 0");
        }

        else if(dec.offset > this.size) {
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

    public void set(int index, float d) {

        this.array[offset + index] = d;
    }

    public void setAll(float val) {

        if((size - offset) >= parallelLength) {
            IntStream.range(offset,size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < size - offset; i++) {
                array[i] = val;
            }
        }

    }

    public void setAll(float val, int size) {

        if(size >= parallelLength) {
            IntStream.range(offset,offset + size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < offset + size; i++) {
                array[i] = val;
            }
        }
    }

    public void copyInto(int size,int offsetSrc, FloatArray dest, int offsetdest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc]);
        }
        else {
            for(int i = 0; i < size; i++) {
                dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc];
            }
        }
    }

    public void copyInto(int size, FloatArray dest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i] = this.array[this.offset + i]);
        }
        else {
            System.arraycopy(this.array, this.offset, dest.array, dest.offset, size);
        }
    }

    public void addIn(int position, float val) {

        this.array[offset + position] += val;
    }

    public void subIn(int position, float val) {

        this.array[offset + position] -= val;
    }

    public void mulIn(int position, float val) {

        this.array[offset + position] *= val;
    }

    public void divIn(int position, float val) {

        this.array[offset + position] /= val;
    }
}
