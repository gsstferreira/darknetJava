package Classes.Arrays;

import java.util.stream.IntStream;

public class IntArray {

    private final int[] array;
    private int offset;

    public IntArray(int[] arr) {

        this.array = arr;
        this.offset = 0;
    }

    public IntArray(int size) {

        this.array = new int[size];
        this.offset = 0;
    }

    public void offset(int off) {
        this.offset += off;

        if(this.offset < 0) {
            throw new IndexOutOfBoundsException("IntArray offset is lesser than 0");
        }

        else if(this.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionArray offset is bigger than buffer length");
        }
    }

    public IntArray offsetNew(int off) {

        IntArray dec = new IntArray(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("IntArray offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
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

    public void put(int index,int d) {

        this.array[index + offset] = d;
    }

    public void setValue(int val) {

        IntStream.range(offset,array.length).forEach(i -> {
            array[i] = val;
        });
    }

    public void setValue(int val, int size) {

        IntStream.range(offset,offset + size).forEach(i -> {
            array[i] = val;
        });
    }

    public int size() {

        return this.array.length - this.offset;
    }

    public void copyInto(int size,int offsetSrc, IntArray dest, int offsetdest) {

        IntStream.range(0,size).parallel().forEach(i -> dest.array[i*offsetdest] = this.array[i*offsetSrc]);
    }

    public void copyInto(int size, IntArray dest) {

        IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i] = this.array[this.offset + i]);
    }
}
