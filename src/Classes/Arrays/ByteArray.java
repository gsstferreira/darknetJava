package Classes.Arrays;

import java.nio.ByteBuffer;
import java.util.stream.IntStream;

public class ByteArray extends ArrayBase{

    private final ByteBuffer buffer;
    private final byte[] array;

    public ByteArray(byte[] arr) {

        this.array = arr;
        this.size = arr.length;
        this.offset = 0;
        this.buffer = ByteBuffer.allocate(8);
    }

    public ByteArray(int size) {

        this.array = new byte[size];
        this.offset = 0;
        this.size = size;
        this.buffer = ByteBuffer.allocate(8);
    }

    public ByteArray offsetNew(int off) {

        ByteArray dec = new ByteArray(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("ByteArray offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("ByteArray offset is bigger than buffer length");
        }

        return dec;
    }

    public byte get(int index) {

        return this.array[index + offset];
    }

    public ByteArray shallowClone() {

        ByteArray buff = new ByteArray(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void set(int index, Byte d) {

        this.array[index + offset] = d;
    }

    public void setAll(byte val) {

        if((size - offset) >= parallelLength) {
            IntStream.range(offset,size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < size - offset; i++) {
                array[i] = val;
            }
        }

    }

    public void setAll(byte val, int size) {

        if(size >= parallelLength) {
            IntStream.range(offset,offset + size).forEach(i -> array[i] = val);
        }
        else {
            for(int i = offset; i < offset + size; i++) {
                array[i] = val;
            }
        }
    }

    public int getNextInt(int startPos) {

        buffer.put(0,this.get(startPos + 3));
        buffer.put(1,this.get(startPos + 2));
        buffer.put(2,this.get(startPos + 1));
        buffer.put(3,this.get(startPos));
        buffer.position(0);

        return buffer.getInt();
    }

    public long getNextLong(int startPos) {

        buffer.put(0,this.get(startPos + 7));
        buffer.put(1,this.get(startPos + 6));
        buffer.put(2,this.get(startPos + 5));
        buffer.put(3,this.get(startPos + 4));
        buffer.put(4,this.get(startPos + 3));
        buffer.put(5,this.get(startPos + 2));
        buffer.put(6,this.get(startPos + 1));
        buffer.put(7,this.get(startPos));
        buffer.position(0);

        return buffer.getLong();
    }

    public float getNextFloat(int startPos) {

        buffer.put(0,this.get(startPos + 3));
        buffer.put(1,this.get(startPos + 2));
        buffer.put(2,this.get(startPos + 1));
        buffer.put(3,this.get(startPos));
        buffer.position(0);

        return buffer.getFloat();
    }

    public void copyInto(int size,int offsetSrc, ByteArray dest, int offsetdest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc]);
        }
        else {
            for(int i = 0; i < size; i++) {
                dest.array[dest.offset + i*offsetdest] = this.array[this.offset + i*offsetSrc];
            }
        }
    }

    public void copyInto(int size, ByteArray dest) {

        if(size >= parallelLength) {
            IntStream.range(0,size).parallel().forEach(i -> dest.array[dest.offset + i] = this.array[this.offset + i]);
        }
        else {
            System.arraycopy(this.array, this.offset, dest.array, dest.offset, size);
        }
    }

    public void addIn(int position, byte val) {

        this.array[offset + position] += val;
    }

    public void subIn(int position, byte val) {

        this.array[offset + position] -= val;
    }

    public void mulIn(int position, byte val) {

        this.array[offset + position] *= val;
    }

    public void divIn(int position, byte val) {

        this.array[offset + position] /= val;
    }
}
