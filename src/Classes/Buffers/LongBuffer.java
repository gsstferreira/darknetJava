package Classes.Buffers;

public class LongBuffer {

    private final long[] array;
    private int offset;

    public LongBuffer(long[] arr) {

        this.array = arr;
        this.offset = 0;
    }

    public LongBuffer(int size) {

        this.array = new long[size];
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

    public LongBuffer offsetNew(int off) {

        LongBuffer dec = new LongBuffer(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("FloatBuffer offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is bigger than buffer length");
        }

        return dec;
    }

    public long get(int index) {

        return this.array[index + offset];
    }

    /**
     * Creates a new LongBuffer object which is a shallow copy of the original object
     * @return LongBuffer object, with reference to the same array
     */
    public LongBuffer shallowClone() {

        LongBuffer buff = new LongBuffer(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void put(int index,long d) {

        this.array[index + offset] = d;
    }

    public int size() {

        return this.array.length - this.offset;
    }
}
