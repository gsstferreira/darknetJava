
import Tools.Detector;
import Tools.GlobalVars;
import com.google.gson.Gson;

public class Main {

    private static final String networkCfgPath = "Res/network.cfg";
    private static final String namesCfgPath = "Res/names.cfg";
    private static final String weightsPath = "Res/weight.weights";

    public static void main(String[] args) {

        String imgPath = "C:/Yolo/darknet/data/trio2.jpg";

        GlobalVars.loadAlphabet();
        GlobalVars.loadNetwork(networkCfgPath,weightsPath,namesCfgPath);

        var x = Detector.runDetector(namesCfgPath,networkCfgPath,weightsPath,imgPath);

        String s = new Gson().toJson(x);

        System.out.println(s);

    }
}
