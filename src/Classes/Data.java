package Classes;

import Tools.GlobalVars;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import java.io.InputStreamReader;
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
            BufferedReader reader;
            if(GlobalVars.isJar) {
                reader = new BufferedReader(new InputStreamReader(Data.class.getResourceAsStream("/" + fileName)));
            }
            else {
                reader = new BufferedReader(new FileReader(fileName));
            }
            String s;

            while((s = reader.readLine()) != null) {
                list.add(s.replace("\n",""));
            }
            reader.close();

            return list;
        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println(String.format("Error trying to load file '%s'.",fileName));
            return null;
        }
    }

    public static List<String> getLabels(String filename) {

        List<String> list = getPaths(filename);

        return list;
    }
}
