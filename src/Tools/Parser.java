package Tools;

import Classes.*;
import Enums.LayerType;
import Enums.LearningRatePolicy;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

public abstract class Parser {

    private static LayerType stringToLayerType(String type) {

        String ss = type.replace("[","").replace("]","");
        return LayerType.getLayerType(ss);
    }

    public static String optionFind(List<KeyValuePair> l , String key) {

        for (KeyValuePair kvp:l) {
            if(kvp.key.equals(key)) {
                kvp.used = true;
                return kvp.value;
            }
        }
        return null;
    }

    public static String optionFindString(List<KeyValuePair> l , String key, String def) {

        String v = optionFind(l, key);
        if(v != null) {
            return v;
        }
        else {
            return def;
        }
    }

    public static int optionFindInt(List<KeyValuePair> l , String key, int def) {

        String v = optionFind(l, key);
        if(v != null) {
            return Integer.parseInt(v);
        }
        else {
            return def;
        }
    }

    public static float optionFindFloat(List<KeyValuePair> l , String key, float def) {

        String v = optionFind(l, key);
        if(v != null) {
            return Float.parseFloat(v.replace(",","."));
        }
        else {
            return def;
        }
    }

    public static List<Section> readCfg(String fileName) {

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            String s;

            List<Section> list = new ArrayList<>();
            Section current = new Section();

            while((s = reader.readLine()) != null) {
                s = s.strip();

                if(!s.isEmpty()) {
                    switch (s.charAt(0)) {
                        case '[':
                            current = new Section();
                            current.options = new ArrayList<>();
                            current.type = s;
                            list.add(current);
                            break;
                        case '#':
                        case ';':
                            break;
                        default:
                            String[] sArr = s.split("=");

                            if(sArr.length == 2 && !sArr[1].isEmpty()) {
                                KeyValuePair kvp = new KeyValuePair(sArr[0],sArr[1]);
                                current.options.add(kvp);
                            }
                            break;
                    }
                }
            }
            reader.close();

            return list;
        }
        catch (Exception e) {
            System.out.println(String.format("Error trying to load file '%s'.",fileName));
            return null;
        }

    }
    
    public static void parseData(String data, FloatBuffer a, int n) {

        if(data != null) {

            String[] values = data.split(",");

            for(int i = 0; i < n && i < values.length; i++) {
                float val = Float.parseFloat(values[i].strip());
                a.put(i,val);
            }
        }
    }

    public static void parseNetworkOptions(List<KeyValuePair> options, Network net) {
        
        net.batch = optionFindInt(options, "batch",1);
        net.learningRate = optionFindFloat(options, "learning_rate", 0.001f);
        net.momentum = optionFindFloat(options, "momentum", 0.9f);
        net.decay = optionFindFloat(options, "decay", 0.0001f);
        int subdivs = optionFindInt(options, "subdivisions",1);
        net.timeSteps = optionFindInt(options, "time_steps",1);
        net.noTruth = optionFindInt(options, "notruth",0);
        net.batch /= subdivs;
        net.batch *= net.timeSteps;
        net.subdivisions = subdivs;
        net.random = optionFindInt(options, "random", 0);

        net.adam = optionFindInt(options, "adam", 0);
        if(net.adam != 0){
            net.B1 = optionFindFloat(options, "B1", 0.9f);
            net.B2 = optionFindFloat(options, "B2", 0.999f);
            net.eps = optionFindFloat(options, "eps", 0.0000001f);
        }

        net.h = optionFindInt(options, "height",0);
        net.w = optionFindInt(options, "width",0);
        net.c = optionFindInt(options, "channels",0);
        net.inputs = optionFindInt(options, "inputs", net.h * net.w * net.c);
        net.maxCrop = optionFindInt(options, "max_crop",net.w*2);
        net.minCrop = optionFindInt(options, "min_crop",net.w);
        net.maxRatio = optionFindFloat(options, "max_ratio", (float) net.maxCrop / net.w);
        net.minRatio = optionFindFloat(options, "min_ratio", (float) net.minCrop / net.w);
        net.center = optionFindInt(options, "center",0);
        net.clip = optionFindFloat(options, "clip", 0);

        net.angle = optionFindFloat(options, "angle", 0);
        net.aspect = optionFindFloat(options, "aspect", 1);
        net.saturation = optionFindFloat(options, "saturation", 1);
        net.exposure = optionFindFloat(options, "exposure", 1);
        net.hue = optionFindFloat(options, "hue", 0);

        if(net.inputs == 0 && net.h == 0 || net.w == 0 || net.c == 0)  {
            ExceptionThrower.InvalidParams("No input parameters supplied");
        }

        String policyS = optionFindString(options, "policy", "constant");
        net.policy = LearningRatePolicy.getLearningRatePolicy(policyS);
        net.burnIn = optionFindInt(options, "burn_in", 0);
        net.power = optionFindFloat(options, "power", 4);

        if(net.policy == LearningRatePolicy.STEP){
            net.step = optionFindInt(options, "step", 1);
            net.scale = optionFindFloat(options, "scale", 1);
        }
        else if (net.policy == LearningRatePolicy.STEPS){
            String l = optionFind(options, "steps");
            String p = optionFind(options, "scales");

            if(l == null || p == null) {
                ExceptionThrower.InvalidParams("STEPS policy must have steps and scales in cfg file");
            }

            String[] lArr = l.split(",");
            String[] pArr = p.split(",");

            int[] steps = new int[lArr.length];
            float[] scales = new float[pArr.length];

            for(int i = 0; i < lArr.length; ++i){

                steps[i] = Integer.parseInt(lArr[i].strip());
                scales[i] = Float.parseFloat(pArr[i].strip());
            }
            net.scales = FloatBuffer.wrap(scales);
            net.steps = IntBuffer.wrap(steps);
            net.numSteps = lArr.length;
        }
        else if (net.policy == LearningRatePolicy.EXP){
            net.gamma = optionFindFloat(options, "gamma", 1);
        }
        else if (net.policy == LearningRatePolicy.SIG){
            net.gamma = optionFindFloat(options, "gamma", 1);
            net.step = optionFindInt(options, "step", 1);
        }

        net.maxBatches = optionFindInt(options, "max_batches", 0);
    }

    public static boolean isNetwork(Section s) {

        return s.type.equals("[net") || s.type.equals("[network");
    }


    

}
