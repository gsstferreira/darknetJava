package Classes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Data {

    public int w;
    public int h;
    public Matrix X;
    public Matrix Y;
    public int shallow;
    public int[] numBoxes;
    public Box[][] boxes;

    public static List<String> getPaths(String fileName) {

        List<String> list = new ArrayList<>();

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            String s;

            while((s = reader.readLine()) != null) {
                list.add(s.replace("\n",""));
            }
            reader.close();

            return list;
        }
        catch (Exception e) {
            System.out.println(String.format("Error trying to load file '%s'.",fileName));
            return null;
        }
    }
}
