package Classes.Arrays;

public abstract class ArrayBase {

    static final int parallelLength = 7500;
    int size;
    int offset;

    public int size() {
         return size - offset;
    }

    public void offset(int off) {

        if(off != 0) {

            this.offset += off;

            if(this.offset >= this.size) {
                this.offset = this.size -1;
            }

            else if(this.offset < 0) {
                throw new IndexOutOfBoundsException("Array offset is lesser than 0");
            }
        }
    }
}
