package Classes.Buffers;

public class ByteBuffer {

    private final byte[] array;
    private int offset;

    private final java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(8);

    public ByteBuffer(byte[] arr) {

        this.array = arr;
        this.offset = 0;
    }

    public ByteBuffer(int size) {

        this.array = new byte[size];
        this.offset = 0;
    }

    public void offset(int off) {
        this.offset += off;

        if(this.offset < 0) {
            throw new IndexOutOfBoundsException("ByteBuffer offset is lesser than 0");
        }
    }

    public ByteBuffer offsetNew(int off) {

        ByteBuffer dec = new ByteBuffer(this.array);

        dec.offset = off + this.offset;

        if(dec.offset < 0) {
            throw new IndexOutOfBoundsException("ByteBuffer offset is lesser than 0");
        }

        else if(dec.offset > this.array.length) {
            throw new IndexOutOfBoundsException("ByteBuffer offset is bigger than buffer length");
        }

        return dec;
    }

    public byte get(int index) {

        return this.array[index + offset];
    }

    /**
     * Creates a new ByteBuffer object which is a shallow copy of the original object
     * @return ByteBuffer object, with reference to the same array
     */
    public ByteBuffer shallowClone() {

        ByteBuffer buff = new ByteBuffer(this.array);
        buff.offset(this.offset);

        return buff;
    }

    public void put(int index,Byte d) {

        this.array[index + offset] = d;
    }

    public int size() {

        return this.array.length - this.offset;
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
}
