import Classes.Image;
import Server.Handlers.RequestHandler;
import Tools.Global;
import Yolo.Enums.ImType;
import Yolo.Setup;
import java.io.*;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class Main {

    private static final int port = 8080;

    public static void main(String[] args) {

        yoloDefault(args);
    }

    private static void yoloDefault(String[] args) {
        Global.displayResult = false;

        for (String s:args) {

            if(s.equals("displayResult")) {
                Global.displayResult = true;
            }

            else if(s.equals("saveResult")) {
                Global.saveResult = true;
            }
        }

        Locale.setDefault(Locale.ENGLISH);

        Setup.initYolo();
        System.gc();

        try {
            System.out.printf("Setting up server at port %d...\n",port);
            ServerSocket serverConnect = new ServerSocket(port);
            System.out.printf("Server ready, listening to port %d.\n",port);

            //noinspection InfiniteLoopStatement
            while(true) {
                new RequestHandler(serverConnect.accept());
            }

        }
        catch (IOException e) {
            System.err.println("Server setup error : " + e.getMessage());
        }
    }

    //extras
    private static final String dataSetPath = "C:/Yolo/training/labels/";
    private static final String outPathCw = "C:/Yolo/training/labels_cw/";
    private static final String outPathCcw = "C:/Yolo/training/labels_ccw/";
    private static final String listOut = "C:/Yolo/training/";
    private static final String outPathRelative = "../training/labels/";
    private static final String outPathRelativeCw = "../training/labels_cw/";
    private static final String outPathRelativeCcw = "../training/labels_ccw/";

    private static final int k_ratio = 10;

    private static void imageRotate(String[] args) {

        try {

            new File(outPathCw).mkdirs();
            new File(outPathCcw).mkdirs();

            File f = new File(dataSetPath);

            String[] names = f.list((dir, name) -> name.contains(".jpg"));

            for (String name:names) {

                String nameTxt = name.replace(".jpg",".txt");

                Image img = Image.loadImageColor(dataSetPath + name,0,0,false);
                Image imgCw = img.rotate90Cw();
                Image imgCcw = img.rotate90Ccw();

                BufferedReader fileReader = new BufferedReader(new FileReader(new File(dataSetPath + nameTxt)));
                List<String> lines = new ArrayList<>();

                String line;
                while ((line = fileReader.readLine()) != null) {
                    lines.add(line);
                }
                fileReader.close();

                FileWriter writerCw = new FileWriter(outPathCw + nameTxt.replace(".txt","_rot_cw.txt"));
                FileWriter writerCcw = new FileWriter(outPathCcw + nameTxt.replace(".txt","_rot_ccw.txt"));

                for (String l:lines) {

                    String[] numbers = l.split(" ");

                    int labelClass = Integer.parseInt(numbers[0]);
                    double centerX = Double.parseDouble(numbers[1]);
                    double centerY = Double.parseDouble(numbers[2]);
                    double widthRelative = Double.parseDouble(numbers[3]);
                    double heightRelative = Double.parseDouble(numbers[4]);

                    int topLeftX = (int) ((centerX - widthRelative/2) * img.w);
                    int topLeftY = (int) ((centerY - heightRelative/2) * img.h);
                    int width = (int) (widthRelative * img.w);
                    int height = (int) (heightRelative * img.h);

                    int newTopCw = img.w - topLeftX - width;
                    int newRightCw = topLeftY + height;
                    int newBotCw = img.w - topLeftX;

                    double midXCw = ((newRightCw + topLeftY)/2.0)/img.w;
                    double midYCw = ((newBotCw + newTopCw)/2.0)/img.h;

                    writerCw.write(String.format(Locale.ENGLISH,"%d %f %f %f %f\n",labelClass,midXCw,midYCw,heightRelative,widthRelative));

                    int newLeftCcw = img.h - topLeftY - height;
                    int newRightCcw = img.h - topLeftY;
                    int newBotCcw = topLeftX + width;

                    double midXCcw = ((newRightCcw + newLeftCcw)/2.0)/img.w;
                    double midYCcw = ((newBotCcw + topLeftX)/2.0)/img.h;

                    writerCcw.write(String.format(Locale.ENGLISH,"%d %f %f %f %f\n",labelClass,midXCcw,midYCcw,heightRelative,widthRelative));
                }
                writerCw.close();
                writerCcw.close();
                imgCw.saveToDisk(outPathCw + name.replace(".jpg","_rot_cw"), ImType.JPG,100);
                imgCcw.saveToDisk(outPathCcw + name.replace(".jpg","_rot_ccw"),ImType.JPG,100);
                System.out.printf("Image '%s' rotated and saved!\n",name);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void imageLabel(String[] args) {

        try {

            File f = new File(dataSetPath);
            File fCw = new File(outPathCw);
            File fCcw = new File(outPathCcw);

            String[] names = f.list((dir, name) -> name.contains(".jpg"));
            String[] namesCw = fCw.list((dir, name) -> name.contains(".jpg"));
            String[] namesCcw = fCcw.list((dir, name) -> name.contains(".jpg"));

            FileWriter writerTrain = new FileWriter(new File(listOut + "train_new.txt"));
            FileWriter writerTest = new FileWriter(new File(listOut + "test_new.txt"));

            int n = 0;

            if(names != null) {
                for(String s:names) {
                    String path = outPathRelative + s + "\n";
                    if(n%k_ratio == 0) {
                        writerTest.write(path);
                    }
                    else {
                        writerTrain.write(path);
                    }
                    n++;
                }
            }

            if(namesCw != null) {
                for(String s:namesCw) {
                    String path = outPathRelativeCw + s + "\n";
                    if(n%k_ratio == 0) {
                        writerTest.write(path);
                    }
                    else {
                        writerTrain.write(path);
                    }
                    n++;
                }
            }

            if(namesCcw != null) {
                for(String s:namesCcw) {
                    String path = outPathRelativeCcw + s + "\n";
                    if(n%k_ratio == 0) {
                        writerTest.write(path);
                    }
                    else {
                        writerTrain.write(path);
                    }
                    n++;
                }
            }

            writerTrain.close();
            writerTest.close();

        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
