package Classes.Arrays;

import java.util.stream.IntStream;

public class IntArray extends ArrayBase{

    private final int[] array;

    public IntArray(int[] arr) {

        this.array = arr;
        this.size = arr.length;
        this.offset = 0;
    }

    public IntArray(int size) {

        this.array = new int[size];
        this.size = size;
        this.offset = 0;
    }

    public IntArray offsetNew(int off) {

        IntArray dec = new IntArray(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("IntArray offset is lesser than 0");
        }

        else if(dec.offset > size) {
            throw new IndexOutOfBoundsException("DetectionArray offset is bigger than buffer length");
        }

        return dec;
    }

    public int get(int index) {

        return this.array[index + offset];
    }

    public IntArray shallowClone() {

        IntArray buff = new IntArray(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void set(int index, int d) {

        this.array[index + offset] = d;
    }

    public void setAll(int val) {

        if((size - offset) >= parallelLength) {
            IntStream.range(offset,size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < size - offset; i++) {
                array[i] = val;
            }
        }

    }

    public void setAll(int val, int size) {

        if(size >= parallelLength) {
            IntStream.range(offset,offset + size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < offset + size; i++) {
                array[i] = val;
            }
        }
    }

    public void copyInto(int size,int offsetSrc, IntArray dest, int offsetdest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc]);
        }
        else {
            for(int i = 0; i < size; i++) {
                dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc];
            }
        }
    }

    public void copyInto(int size, IntArray dest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i] = this.array[this.offset + i]);
        }
        else {
            System.arraycopy(this.array, this.offset, dest.array, dest.offset, size);
        }
    }

    public void addIn(int position, int val) {

        this.array[offset + position] += val;
    }

    public void subIn(int position, int val) {

        this.array[offset + position] -= val;
    }

    public void mulIn(int position, int val) {

        this.array[offset + position] *= val;
    }

    public void divIn(int position, int val) {

        this.array[offset + position] /= val;
    }
}
