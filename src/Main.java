import Classes.Image;
import Classes.Network;
import Enums.ImType;
import Tools.BufferUtil;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        IntBuffer b = IntBuffer.allocate(10);

        for(int i = 0; i < 10; i++) {
            b.put(i,3*i);
        }

        System.out.println(Arrays.toString(b.array()));

        for(int i = 0; i < 10; i++) {

            b.put(i,b.get(i) *2);
        }

        System.out.println(Arrays.toString(b.array()));
    }
}
