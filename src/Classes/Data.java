package Classes;

import Tools.Global;
import Tools.Util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Data {

    public int w;
    public int h;
//    public Matrix X;
//    public Matrix Y;
//    public int shallow;
//    public int[] numBoxes;
//    public Box[][] boxes;

    public static String[] getPaths(String fileName) {

        try {
            BufferedReader reader;
            if(Global.isJar) {
                reader = new BufferedReader(new InputStreamReader(Data.class.getResourceAsStream("/" + fileName)));
            }
            else {
                reader = new BufferedReader(new FileReader(fileName));
            }

            StringBuilder concat = new StringBuilder();
            String s;

            while((s = reader.readLine()) != null) {
                concat.append(Util.strip(s)).append("\n\r");
            }
            reader.close();

            return Util.strip(concat.toString()).split("\n\r");
        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println(String.format("Error trying to load file '%s'.",fileName));
            return null;
        }
    }

//    public static List<String> getLabels(String filename) {
//
//        List<String> list = getPaths(filename);
//
//        return list;
//    }
}
