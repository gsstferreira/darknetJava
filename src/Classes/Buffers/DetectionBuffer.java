package Classes.Buffers;

import Classes.Detection;

public class DetectionBuffer {

    private Detection[] array;
    private int offset;

    public DetectionBuffer(Detection[] arr) {

        this.array = arr;
        this.offset = 0;
    }

    public DetectionBuffer(int size) {

        this.array = new Detection[size];
        this.offset = 0;
    }

    public void offset(int off) {
        this.offset += off;

        if(this.offset < 0) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is lesser than 0");
        }
    }

    public DetectionBuffer offsetNew(int off) {

        DetectionBuffer dec = new DetectionBuffer(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is bigger than buffer length");
        }

        return dec;
    }

    public Detection get(int index) {

        return this.array[index + offset];
    }

    /**
     * Creates a new DetectionBuffer object which is a shallow copy of the original object
     * @return DetectionBuffer object, with reference to the same array
     */
    public DetectionBuffer shallowClone() {

        DetectionBuffer buff = new DetectionBuffer(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void put(int index,Detection d) {

        this.array[index + offset] = d;
    }

    public int size() {

        return this.array.length - this.offset;
    }

}
