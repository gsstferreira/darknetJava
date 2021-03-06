package Classes.Arrays;

import Classes.Detection;

public class DetectionArray extends ArrayBase{

    private final Detection[] array;
    private int offset;

    public DetectionArray(Detection[] arr) {

        this.array = arr;
        this.size = arr.length;
        this.offset = 0;
    }

    public DetectionArray(int size) {

        this.array = new Detection[size];
        this.size = size;
        this.offset = 0;
    }

    public DetectionArray offsetNew(int off) {

        DetectionArray dec = new DetectionArray(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("DetectionArray offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionArray offset is bigger than buffer length");
        }

        return dec;
    }

    public Detection get(int index) {

        return this.array[index + offset];
    }

    public DetectionArray shallowClone() {

        DetectionArray buff = new DetectionArray(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void set(int index, Detection d) {

        this.array[index + offset] = d;
    }

}
