import Classes.Image;
import Classes.Section;
import Enums.ImType;
import Tools.BufferUtils;
import Tools.Parser;
import Tools.Rand;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;

public class Main {

    public static void main(String[] args) {

        Image img = Image.loadImage("C:/Users/gferreira/Pictures/pepsis.png",630,354,3);

        Image img2 = img.resizeImage(315,202);

        img.saveToDisk("C:/Users/gferreira/Pictures/pepsisSame", ImType.JPG,100);

    }
}
