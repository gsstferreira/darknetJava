package Classes.Buffers;

public class IntBuffer {

    private int[] array;
    private int offset;

    public IntBuffer(int[] arr) {

        this.array = arr;
        this.offset = 0;
    }

    public IntBuffer(int size) {

        this.array = new int[size];
        this.offset = 0;
    }

    public void offset(int off) {
        this.offset += off;

        if(this.offset < 0) {
            throw new IndexOutOfBoundsException("IntBuffer offset is lesser than 0");
        }

        else if(this.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is bigger than buffer length");
        }
    }

    public IntBuffer offsetNew(int off) {

        var dec = new IntBuffer(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("IntBuffer offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("DetectionBuffer offset is bigger than buffer length");
        }

        return dec;
    }

    public int get(int index) {

        return this.array[index + offset];
    }

    public void put(int index,int d) {

        this.array[index + offset] = d;
    }

    public int size() {

        return this.array.length - this.offset;
    }

}
