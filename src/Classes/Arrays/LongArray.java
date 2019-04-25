package Classes.Arrays;

import java.util.stream.IntStream;

public class LongArray extends ArrayBase{

    private final long[] array;

    public LongArray(long[] arr) {

        this.array = arr;
        this.size = arr.length;
        this.offset = 0;
    }

    public LongArray(int size) {

        this.array = new long[size];
        this.size = size;
        this.offset = 0;
    }

    public LongArray offsetNew(int off) {

        LongArray dec = new LongArray(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("FloatArray offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionArray offset is bigger than buffer length");
        }

        return dec;
    }

    public long get(int index) {

        return this.array[index + offset];
    }

    public LongArray shallowClone() {

        LongArray buff = new LongArray(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void set(int index, long d) {

        this.array[index + offset] = d;
    }

    public void setAll(long val) {

        if((size - offset) >= parallelLength) {
            IntStream.range(offset,size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < size - offset; i++) {
                array[i] = val;
            }
        }

    }

    public void setAll(long val, int size) {

        if(size >= parallelLength) {
            IntStream.range(offset,offset + size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < offset + size; i++) {
                array[i] = val;
            }
        }
    }

    public void copyInto(int size,int offsetSrc, LongArray dest, int offsetdest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc]);
        }
        else {
            for(int i = 0; i < size; i++) {
                dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc];
            }
        }
    }

    public void copyInto(int size, LongArray dest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i] = this.array[this.offset + i]);
        }
        else {
            System.arraycopy(this.array, this.offset, dest.array, dest.offset, size);
        }
    }

    public void addIn(int position, long val) {

        this.array[offset + position] += val;
    }

    public void subIn(int position, long val) {

        this.array[offset + position] -= val;
    }

    public void mulIn(int position, long val) {

        this.array[offset + position] *= val;
    }

    public void divIn(int position, long val) {

        this.array[offset + position] /= val;
    }
}
