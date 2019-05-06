import Classes.Image;
import Yolo.Enums.ImType;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ImageRotate {

    private static final String dataSetPath = "C:/Yolo/training/labels/";
    private static final String outPathCw = "C:/Yolo/training/labels_cw/";
    private static final String outPathCcw = "C:/Yolo/training/labels_ccw/";

    public static void main(String[] args) {

//        try {
//
//            new File(outPathCw).mkdirs();
//            new File(outPathCcw).mkdirs();
//
//            File f = new File(dataSetPath);
//
//            String[] names = f.list((dir, name) -> name.contains(".jpg"));
//
//            for (String name:names) {
//
//                String nameTxt = name.replace(".jpg",".txt");
//
//                Image img = Image.loadImageColor(dataSetPath + name,0,0,false);
//                Image imgCw = img.rotate90Cw();
//                Image imgCcw = img.rotate90Ccw();
//
//                BufferedReader fileReader = new BufferedReader(new FileReader(new File(dataSetPath + nameTxt)));
//                List<String> lines = new ArrayList<>();
//
//                String line;
//                while ((line = fileReader.readLine()) != null) {
//                    lines.add(line);
//                }
//                fileReader.close();
//
//                FileWriter writerCw = new FileWriter(outPathCw + nameTxt.replace(".txt","_rot_cw.txt"));
//                FileWriter writerCcw = new FileWriter(outPathCcw + nameTxt.replace(".txt","_rot_ccw.txt"));
//
//                for (String l:lines) {
//
//                    String[] numbers = l.split(" ");
//
//                    int labelClass = Integer.parseInt(numbers[0]);
//                    double centerX = Double.parseDouble(numbers[1]);
//                    double centerY = Double.parseDouble(numbers[2]);
//                    double widthRelative = Double.parseDouble(numbers[3]);
//                    double heightRelative = Double.parseDouble(numbers[4]);
//
//                    int topLeftX = (int) ((centerX - widthRelative/2) * img.w);
//                    int topLeftY = (int) ((centerY - heightRelative/2) * img.h);
//                    int width = (int) (widthRelative * img.w);
//                    int height = (int) (heightRelative * img.h);
//
//                    int newTopCw = img.w - topLeftX - width;
//                    int newRightCw = topLeftY + height;
//                    int newBotCw = img.w - topLeftX;
//
//                    double midXCw = ((newRightCw + topLeftY)/2.0)/img.w;
//                    double midYCw = ((newBotCw + newTopCw)/2.0)/img.h;
//
//                    writerCw.write(String.format(Locale.ENGLISH,"%d %f %f %f %f\n",labelClass,midXCw,midYCw,heightRelative,widthRelative));
//
//                    int newLeftCcw = img.h - topLeftY - height;
//                    int newRightCcw = img.h - topLeftY;
//                    int newBotCcw = topLeftX + width;
//
//                    double midXCcw = ((newRightCcw + newLeftCcw)/2.0)/img.w;
//                    double midYCcw = ((newBotCcw + topLeftX)/2.0)/img.h;
//
//                    writerCcw.write(String.format(Locale.ENGLISH,"%d %f %f %f %f\n",labelClass,midXCcw,midYCcw,heightRelative,widthRelative));
//                }
//                writerCw.close();
//                writerCcw.close();
//                imgCw.saveToDisk(outPathCw + name.replace(".jpg","_rot_cw"),ImType.JPG,100);
//                imgCcw.saveToDisk(outPathCcw + name.replace(".jpg","_rot_ccw"),ImType.JPG,100);
//                System.out.printf("Image '%s' rotated and saved!\n",name);
//            }
//        }
//        catch (Exception e) {
//            e.printStackTrace();
//        }
    }
}