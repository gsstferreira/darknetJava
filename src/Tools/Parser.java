package Tools;

import Classes.*;
import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Enums.Activation;
import Enums.CostType;
import Enums.LayerType;
import Enums.LearningRatePolicy;
import Layers.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static Enums.LayerType.*;

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
            if(def != null && def.isEmpty()) {
                return null;
            }
            else {
                return def;
            }
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

            float f = Float.parseFloat(v.replace(",","."));

            return f;
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

    public static List<KeyValuePair> readDataCfg(String fileName) {

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            String s;

            List<KeyValuePair> list = new ArrayList<>();

            while((s = reader.readLine()) != null) {
                s = s.strip();

                if(!s.isEmpty()) {
                    switch (s.charAt(0)) {
                        case '#':
                        case ';':
                            break;
                        default:
                            String[] sArr = s.split("=");

                            if(sArr.length == 2 && !sArr[1].isEmpty()) {
                                KeyValuePair kvp = new KeyValuePair(sArr[0].strip(),sArr[1].strip());
                                list.add(kvp);
                            }
                            break;
                    }
                }
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
    
    public static void parseData(String data, FloatBuffer a, int n) {

        if(data != null) {

            String[] values = data.split(",");

            for(int i = 0; i < n && i < values.length; i++) {
                float val = Float.parseFloat(values[i].strip());
                a.put(i,val);
            }
        }
    }

    public static void parseNetOptions(List<KeyValuePair> options, Network net) {
        
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

        if(net.inputs == 0 && (net.h == 0 || net.w == 0 || net.c == 0))  {
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
            net.scales = new FloatBuffer(scales);
            net.steps = new IntBuffer(steps);
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

        return s.type.equals("[net]") || s.type.equals("[network]");
    }

    public static LocalLayer parseLocal(List<KeyValuePair> options, SizeParams params) {

        int n = optionFindInt(options, "filters",1);
        int size = optionFindInt(options, "size",1);
        int stride = optionFindInt(options, "stride",1);
        int pad = optionFindInt(options, "pad",0);
        String activation_s = optionFindString(options, "activation", "logistic");
        Activation activation = Activation.getActivation(activation_s);

        int batch,h,w,c;
        h = params.h;
        w = params.w;
        c = params.c;
        batch=params.batch;

        if(h == 0 || w == 0 || c == 0) {
            ExceptionThrower.InvalidParams("Layer before local Layer must output image.");
        }

        return new LocalLayer(batch,h,w,c,n,size,stride,pad,activation);
    }

    public static DeconvolutionalLayer parseDeconvolutional(List<KeyValuePair> options, SizeParams params) {

        int n = optionFindInt(options, "filters",1);
        int size = optionFindInt(options, "size",1);
        int stride = optionFindInt(options, "stride",1);

        String activation_s = optionFindString(options, "activation", "logistic");
        Activation activation = Activation.getActivation(activation_s);

        int batch,h,w,c;
        h = params.h;
        w = params.w;
        c = params.c;
        batch=params.batch;
        if(h == 0 || w == 0 || c == 0) {
            ExceptionThrower.InvalidParams("Layer before deconvolutional Layer must output image.");
        }
        int batch_normalize = optionFindInt(options, "batch_normalize", 0);
        int pad = optionFindInt(options, "pad",0);
        int padding = optionFindInt(options, "padding",0);
        if(pad != 0) {
            padding = size/2;
        }

        return new DeconvolutionalLayer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net.adam);
    }

    public static ConvolutionalLayer parseConvolutional(List<KeyValuePair> options, SizeParams params) {

        int n = optionFindInt(options, "filters",1);
        int size = optionFindInt(options, "size",1);
        int stride = optionFindInt(options, "stride",1);
        int pad = optionFindInt(options, "pad",0);
        int padding = optionFindInt(options, "padding",0);
        int groups = optionFindInt(options, "groups", 1);
        if(pad != 0) {
            padding = size/2;
        }

        String activation_s = optionFindString(options, "activation", "logistic");
        Activation activation = Activation.getActivation(activation_s);

        int batch,h,w,c;
        h = params.h;
        w = params.w;
        c = params.c;
        batch = params.batch;

        if(h == 0 || w == 0 || c == 0) ExceptionThrower.InvalidParams("Layer before convolutional Layer must output image.");
        int batch_normalize = optionFindInt(options, "batch_normalize", 0);
        int binary = optionFindInt(options, "binary", 0);
        int xnor = optionFindInt(options, "xnor", 0);

        var layer = new ConvolutionalLayer(batch,h,w,c,n,groups,size,stride,padding,activation, batch_normalize, binary, xnor, params.net.adam);
        layer.flipped = optionFindInt(options, "flipped", 0);
        layer.dot = optionFindFloat(options, "dot", 0);

        return layer;
    }

    public static CrnnLayer parseCrnn(List<KeyValuePair> options, SizeParams params) {

        int output_filters = optionFindInt(options, "output_filters",1);
        int hidden_filters = optionFindInt(options, "hidden_filters",1);
        String activation_s = optionFindString(options, "activation", "logistic");
        Activation activation = Activation.getActivation(activation_s);
        int batch_normalize = optionFindInt(options, "batch_normalize", 0);

        var Layer = new CrnnLayer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize);
        Layer.shortcut = optionFindInt(options, "shortcut", 0);

        return Layer;
    }

    public static RnnLayer parseRnn(List<KeyValuePair> options, SizeParams params) {

        int output = optionFindInt(options, "output",1);
        String activation_s = optionFindString(options, "activation", "logistic");
        Activation activation = Activation.getActivation(activation_s);
        int batch_normalize = optionFindInt(options, "batch_normalize", 0);

        var l = new RnnLayer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net.adam);
        l.shortcut = optionFindInt(options, "shortcut", 0);

        return l;
    }

    public static GruLayer parseGru(List<KeyValuePair> options, SizeParams params) {

        int output = optionFindInt(options, "output",1);
        int batch_normalize = optionFindInt(options, "batch_normalize", 0);

        var l = new GruLayer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net.adam);
        l.tanh = optionFindInt(options, "tanh", 0);

        return l;
    }

    public static LstmLayer parseLstm(List<KeyValuePair> options, SizeParams params) {

        int output = optionFindInt(options, "output", 1);
        int batch_normalize = optionFindInt(options, "batch_normalize", 0);

        return new LstmLayer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net.adam);
    }

    public static ConnectedLayer parseConnected(List<KeyValuePair> options, SizeParams params) {

        int output = optionFindInt(options, "output",1);
        String activation_s = optionFindString(options, "activation", "logistic");
        Activation activation = Activation.getActivation(activation_s);
        int batch_normalize = optionFindInt(options, "batch_normalize", 0);

        return new ConnectedLayer(params.batch, params.inputs, output, activation, batch_normalize, params.net.adam);
    }

    public static SoftmaxLayer parseSoftmax(List<KeyValuePair> options, SizeParams params) {

        int groups = optionFindInt(options, "groups",1);
        var l = new SoftmaxLayer(params.batch, params.inputs, groups);
        l.temperature = optionFindFloat(options, "temperature", 1);
        String tree_file = optionFindString(options, "tree", "");
        if (tree_file != null) {
            l.softmaxTree = Tree.readTree(tree_file);
        }
        l.w = params.w;
        l.h = params.h;
        l.c = params.c;
        l.spatial = (int) optionFindFloat(options, "spatial", 0);
        l.noloss =  optionFindInt(options, "noloss", 0);
        return l;
    }

    public static IntBuffer parseYoloMask(String a, IntBuffer num) {

        IntBuffer mask = null;

        if(a != null){

            String[] sp = a.strip().split(",");
            mask = new IntBuffer(sp.length);

            for(int i = 0; i < sp.length; i++) {
                mask.put(i,Integer.parseInt(sp[i]));
            }

            num.put(0,sp.length);
        }
        return mask;
    }

    public static YoloLayer parseYolo(List<KeyValuePair> options, SizeParams params) {

        int classes = optionFindInt(options, "classes", 20);
        int total = optionFindInt(options, "num", 1);
        int num = total;

        String a = optionFindString(options, "mask", "");
        IntBuffer ib = new IntBuffer(1);
        ib.put(0,num);

        IntBuffer mask = parseYoloMask(a, ib);
        num = ib.get(0);

        var l = new YoloLayer(params.batch, params.w, params.h, num, total, mask, classes);
        assert(l.outputs == params.inputs);

        l.maxBoxes = optionFindInt(options, "max",90);
        l.jitter = optionFindFloat(options, "jitter", .2f);

        l.ignoreThresh = optionFindFloat(options, "ignore_thresh", .5f);
        l.truthThresh = optionFindFloat(options, "truth_thresh", 1);
        l.random = optionFindInt(options, "random", 0);

        String map_file = optionFindString(options, "map", "");
        if (map_file != null) {
            l.map = Util.readMap(map_file);
        }

        a = optionFindString(options, "anchors", "");

        if(a != null){
            
            String[] sp = a.strip().split(",");

            for(int i = 0; i < sp.length; i++) {
                l.biases.put(i,Float.parseFloat(sp[i]));
            }
        }
        return l;
    }

    public static IsegLayer parseIseg(List<KeyValuePair> options, SizeParams params) {
        
        int classes = optionFindInt(options, "classes", 20);
        int ids = optionFindInt(options, "ids", 32);
        var l = new IsegLayer(params.batch, params.w, params.h, classes, ids);
        assert(l.outputs == params.inputs);
        return l;
    }

    public static RegionLayer parseRegion(List<KeyValuePair> options, SizeParams params) {
        
        int coords = optionFindInt(options, "coords", 4);
        int classes = optionFindInt(options, "classes", 20);
        int num = optionFindInt(options, "num", 1);

        var l = new RegionLayer(params.batch, params.w, params.h, num, classes, coords);
        assert(l.outputs == params.inputs);

        l.log = optionFindInt(options, "log", 0);
        l.sqrt = optionFindInt(options, "sqrt", 0);

        l.softmax = optionFindInt(options, "softmax", 0);
        l.background = optionFindInt(options, "background", 0);
        l.maxBoxes = optionFindInt(options, "max",30);
        l.jitter = optionFindFloat(options, "jitter", .2f);
        l.rescore = optionFindInt(options, "rescore",0);

        l.thresh = optionFindFloat(options, "thresh", .5f);
        l.classfix = optionFindInt(options, "classfix", 0);
        l.absolute = optionFindInt(options, "absolute", 0);
        l.random = optionFindInt(options, "random", 0);

        l.coordScale = optionFindFloat(options, "coord_scale", 1);
        l.objectScale = optionFindFloat(options, "object_scale", 1);
        l.noobjectScale = optionFindFloat(options, "noobject_scale", 1);
        l.maskScale = optionFindFloat(options, "mask_scale", 1);
        l.classScale = optionFindFloat(options, "class_scale", 1);
        l.biasMatch = optionFindInt(options, "bias_match",0);

        String tree_file = optionFindString(options, "tree", "");
        if (tree_file != null) {
            l.softmaxTree = Tree.readTree(tree_file);
        }
        String map_file = optionFindString(options, "map", "");
        if (map_file != null) {
            l.map = Util.readMap(map_file);
        }

        String a = optionFindString(options, "anchors", "");
        
        if(a != null){
            
            String[] sp = a.strip().split(",");
            
            for(int i = 0; i < sp.length; i++) {
                l.biases.put(i, Float.parseFloat(sp[i]));
            }
        }
        return l;
    }

    public static DetectionLayer parseDetection(List<KeyValuePair> options, SizeParams params) {
        
        int coords = optionFindInt(options, "coords", 1);
        int classes = optionFindInt(options, "classes", 1);
        int rescore = optionFindInt(options, "rescore", 0);
        int num = optionFindInt(options, "num", 1);
        int side = optionFindInt(options, "side", 7);
        var Layer = new DetectionLayer(params.batch, params.inputs, num, side, classes, coords, rescore);

        Layer.softmax = optionFindInt(options, "softmax", 0);
        Layer.sqrt = optionFindInt(options, "sqrt", 0);

        Layer.maxBoxes = optionFindInt(options, "max",90);
        Layer.coordScale = optionFindFloat(options, "coord_scale", 1);
        Layer.forced = optionFindInt(options, "forced", 0);
        Layer.objectScale = optionFindFloat(options, "object_scale", 1);
        Layer.noobjectScale = optionFindFloat(options, "noobject_scale", 1);
        Layer.classScale = optionFindFloat(options, "class_scale", 1);
        Layer.jitter = optionFindFloat(options, "jitter", .2f);
        Layer.random = optionFindInt(options, "random", 0);
        Layer.reorg = optionFindInt(options, "reorg", 0);
        return Layer;
    }

    public static CostLayer parseCost(List<KeyValuePair> options, SizeParams params) {

        String type_s = optionFindString(options, "type", "sse");
        CostType type = CostType.getCostType(type_s);
        float scale = optionFindFloat(options, "scale",1);
        var Layer = new CostLayer(params.batch, params.inputs, type, scale);
        Layer.ratio =  optionFindFloat(options, "ratio",0);
        Layer.noobjectScale =  optionFindFloat(options, "noobj", 1);
        Layer.thresh =  optionFindFloat(options, "thresh",0);
        return Layer;
    }

    public static CropLayer parseCrop(List<KeyValuePair> options, SizeParams params) {
        
        int crop_height = optionFindInt(options, "crop_height",1);
        int crop_width = optionFindInt(options, "crop_width",1);
        int flip = optionFindInt(options, "flip",0);
        float angle = optionFindFloat(options, "angle",0);
        float saturation = optionFindFloat(options, "saturation",1);
        float exposure = optionFindFloat(options, "exposure",1);

        int batch,h,w,c;
        h = params.h;
        w = params.w;
        c = params.c;
        batch=params.batch;
        if(h == 0 || w == 0 || c == 0) ExceptionThrower.InvalidParams("Layer before crop Layer must output image.");

        int noadjust = optionFindInt(options, "noadjust",0);

        var l = new CropLayer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
        l.shift = optionFindFloat(options, "shift", 0);
        l.noAdjust = noadjust;
        return l;
    }

    public static ReorgLayer parseReorg(List<KeyValuePair> options, SizeParams params) {
        
        int stride = optionFindInt(options, "stride",1);
        int reverse = optionFindInt(options, "reverse",0);
        int flatten = optionFindInt(options, "flatten",0);
        int extra = optionFindInt(options, "extra",0);

        int batch,h,w,c;
        h = params.h;
        w = params.w;
        c = params.c;
        batch=params.batch;
        if(h == 0 || w == 0 || c == 0) {
            ExceptionThrower.InvalidParams("Layer before reorg Layer must output image.");
        }

        return new ReorgLayer(batch,w,h,c,stride,reverse, flatten, extra);
    }

    public static MaxpoolLayer parseMaxpool(List<KeyValuePair> options, SizeParams params) {
        
        int stride = optionFindInt(options, "stride",1);
        int size = optionFindInt(options, "size",stride);
        int padding = optionFindInt(options, "padding", size-1);

        int batch,h,w,c;
        h = params.h;
        w = params.w;
        c = params.c;
        batch=params.batch;
        if(h == 0 || w == 0 || c == 0) {
            ExceptionThrower.InvalidParams("Layer before maxpool Layer must output image.");
        }

        return new MaxpoolLayer(batch,h,w,c,size,stride,padding);
    }

    public static AvgPoolLayer parseAvgpool(List<KeyValuePair> options, SizeParams params) {
        
        int batch,w,h,c;
        w = params.w;
        h = params.h;
        c = params.c;
        batch=params.batch;
        if(h == 0 || w == 0 || c == 0) {
            ExceptionThrower.InvalidParams("Layer before avgpool Layer must output image.");
        }

        var Layer = new AvgPoolLayer(batch,w,h,c);
        return Layer;
    }

    public static DropoutLayer parse_dropout(List<KeyValuePair> options, SizeParams params) {
        
        float probability = optionFindFloat(options, "probability", .5f);
        var Layer = new DropoutLayer(params.batch, params.inputs, probability);
        Layer.outW = params.w;
        Layer.outH = params.h;
        Layer.outC = params.c;
        return Layer;
    }

    public static NormalizationLayer parseNormalization(List<KeyValuePair> options, SizeParams params) {
        
        float alpha = optionFindFloat(options, "alpha", .0001f);
        float beta =  optionFindFloat(options, "beta" , .75f);
        float kappa = optionFindFloat(options, "kappa", 1);
        int size = optionFindInt(options, "size", 5);
        
        return new NormalizationLayer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    }

    public static BatchnormLayer parseBatchnorm(List<KeyValuePair> options, SizeParams params) {
        
        return new BatchnormLayer(params.batch, params.w, params.h, params.c);
    }

    public static ShortcutLayer parseShortcut(List<KeyValuePair> options, SizeParams params, Network net) {
        
        String l = optionFind(options, "from");
        int index = Integer.parseInt(l);
        if(index < 0) {
            index = params.index + index;
        }

        int batch = params.batch;
        Layer from = net.layers[index];

        var s = new ShortcutLayer(batch, index, params.w, params.h, params.c, from.outW, from.outH, from.outC);

        String activation_s = optionFindString(options, "activation", "linear");
        Activation activation = Activation.getActivation(activation_s);
        s.activation = activation;
        s.alpha = optionFindFloat(options, "alpha", 1);
        s.beta = optionFindFloat(options, "beta", 1);
        return s;
    }

    public static L2NormLayer parseL2Norm(List<KeyValuePair> options, SizeParams params) {

        var l = new L2NormLayer(params.batch, params.inputs);
        l.h = l.outH = params.h;
        l.w = l.outW = params.w;
        l.c = l.outC = params.c;
        return l;
    }

    public static LogisticLayer parseLogistic(List<KeyValuePair> options, SizeParams params) {

        var l = new LogisticLayer(params.batch, params.inputs);
        l.h = l.outH = params.h;
        l.w = l.outW = params.w;
        l.c = l.outC = params.c;
        return l;
    }

    public static ActivationLayer parseActivation(List<KeyValuePair> options, SizeParams params) {

        String activation_s = optionFindString(options, "activation", "linear");
        Activation activation = Activation.getActivation(activation_s);

        var l = new ActivationLayer(params.batch, params.inputs, activation);

        l.h = l.outH = params.h;
        l.w = l.outW = params.w;
        l.c = l.outC = params.c;

        return l;
    }

    public static UpsampleLayer parseUpsample(List<KeyValuePair> options, SizeParams params, Network net) {

        int stride = optionFindInt(options, "stride",2);
        var l = new UpsampleLayer(params.batch, params.w, params.h, params.c, stride);
        l.scale = optionFindFloat(options, "scale", 1);
        return l;
    }

    public static RouteLayer parseRoute(List<KeyValuePair> options, SizeParams params, Network net) {

        String l = optionFind(options, "layers");

        if(l == null) {
            ExceptionThrower.InvalidParams("Route Layer must specify input Layers");
        }

        String[] sp = l.strip().split(",");

        IntBuffer layers = new IntBuffer(sp.length);
        IntBuffer sizes = new IntBuffer(sp.length);

        for(int i = 0; i < sp.length; i++) {
            int index = Integer.parseInt(sp[i].strip());

            if(index < 0) {
                index += params.index;
            }
            layers.put(i,index);
            sizes.put(i,net.layers[index].outputs);
        }

        int batch = params.batch;
        var layer = new RouteLayer(batch, sp.length, layers, sizes);

        Layer first = net.layers[layers.get(0)];
        layer.outW = first.outW;
        layer.outH = first.outH;
        layer.outC = first.outC;

        for(int i = 1; i < sp.length; ++i){

            int index = layers.get(i);
            Layer next = net.layers[index];

            if(next.outW == first.outW && next.outH == first.outH){
                layer.outC += next.outC;
            }
            else{
                layer.outH = 0;
                layer.outW = 0;
                layer.outC = 0;
            }
        }
        return layer;
    }

    public static Network parseNetworkCfg(String filename) {

        var listSections = readCfg(filename);
        
        assert listSections != null;
        Section section = listSections.get(0);

        if(section == null) {
            ExceptionThrower.InvalidParams("Config file has no sections");
        }

        Network net = new Network(listSections.size() - 1);
        SizeParams params = new SizeParams();

        var options = section.options;

        if(!isNetwork(section)){
            ExceptionThrower.InvalidParams("First section must be [net] or [network]");
        }

        parseNetOptions(options, net);

        params.h = net.h;
        params.w = net.w;
        params.c = net.c;
        params.inputs = net.inputs;
        params.batch = net.batch;
        params.time_steps = net.timeSteps;
        params.net = net;

        long workspace_size = 0;

        for(int i = 1; i < listSections.size(); i++) {
            section = listSections.get(i);


            params.index = i - 1;
            options = section.options;

            Layer l = new Layer();
            LayerType lt = LayerType.getLayerType(section.type);
            
            if(lt == CONVOLUTIONAL){
                l = parseConvolutional(options, params);
            }else if(lt == DECONVOLUTIONAL){
                l = parseDeconvolutional(options, params);
            }else if(lt == LOCAL){
                l = parseLocal(options, params);
            }else if(lt == ACTIVE){
                l = parseActivation(options, params);
            }else if(lt == LOGXENT){
                l = parseLogistic(options, params);
            }else if(lt == L2NORM){
                l = parseL2Norm(options, params);
            }else if(lt == RNN){
                l = parseRnn(options, params);
            }else if(lt == GRU){
                l = parseGru(options, params);
            }else if (lt == LSTM) {
                l = parseLstm(options, params);
            }else if(lt == CRNN){
                l = parseCrnn(options, params);
            }else if(lt == CONNECTED){
                l = parseConnected(options, params);
            }else if(lt == CROP){
                l = parseCrop(options, params);
            }else if(lt == COST){
                l = parseCost(options, params);
            }else if(lt == REGION){
                l = parseRegion(options, params);
            }else if(lt == YOLO){
                l = parseYolo(options, params);
            }else if(lt == ISEG){
                l = parseIseg(options, params);
            }else if(lt == DETECTION){
                l = parseDetection(options, params);
            }else if(lt == SOFTMAX){
                l = parseSoftmax(options, params);
                net.hierarchy = l.softmaxTree;
            }else if(lt == NORMALIZATION){
                l = parseNormalization(options, params);
            }else if(lt == BATCHNORM){
                l = parseBatchnorm(options, params);
            }else if(lt == MAXPOOL){
                l = parseMaxpool(options, params);
            }else if(lt == REORG){
                l = parseReorg(options, params);
            }else if(lt == AVGPOOL){
                l = parseAvgpool(options, params);
            }else if(lt == ROUTE){
                l = parseRoute(options, params, net);
            }else if(lt == UPSAMPLE){
                l = parseUpsample(options, params, net);
            }else if(lt == SHORTCUT){
                l = parseShortcut(options, params, net);
            }else if(lt == DROPOUT){
                l = parse_dropout(options, params);
                l.output = net.layers[i - 2].output;
                l.delta = net.layers[i - 2].delta;

            }else{
                System.out.println(String.format("Parser:parseNetworkCfg - Layer type not recognized: '%s'\n", section.type));
            }
            
            l.clip = net.clip;
            l.truth = optionFindInt(options, "truth", 0);
            l.onlyforward = optionFindInt(options, "onlyforward", 0);
            l.stopbackward = optionFindInt(options, "stopbackward", 0);
            l.dontsave = optionFindInt(options, "dontsave", 0);
            l.dontload = optionFindInt(options, "dontload", 0);
            l.numload = optionFindInt(options, "numload", 0);
            l.dontloadscales = optionFindInt(options, "dontloadscales", 0);
            l.learningRateScale = optionFindFloat(options, "learning_rate", 1);
            l.smooth = optionFindFloat(options, "smooth", 0);

            net.layers[i - 1] = l;
            if (l.workspaceSize > workspace_size) {
                workspace_size = l.workspaceSize;
            }

            params.h = l.outH;
            params.w = l.outW;
            params.c = l.outC;
            params.inputs = l.outputs;

        }


        Layer out = net.getNetworkOutputLayer();
        net.outputs = out.outputs;
        net.truths = out.outputs;

        if(net.layers[net.n-1].truths != 0) {
            net.truths = net.layers[net.n-1].truths;
        }
        net.output = out.output;

        net.input = new FloatBuffer(net.inputs*net.batch);
        net.truth = new FloatBuffer(net.truths*net.batch);

        if(workspace_size != 0){

            net.workspace = new FloatBuffer((int)workspace_size);

        }
        return net;
    }

    public static void saveConvolutionalWeightsBinary(Layer l, BufferedOutputStream file) {

        try {
            ConvolutionalLayer.binarizeWeights(l.weights, l.n, l.c*l.size*l.size, l.binaryWeights);
            int size = l.c*l.size*l.size;
            int i, j, k;

            for(i = 0; i < l.n; i++) {
                byte[] buf = Util.toByteArray(l.biases.get(i));
                file.write(buf);
            }

            if (l.batchNormalize != 0){

                for(i = 0; i < l.n; i++) {
                    byte[] buf = Util.toByteArray(l.scales.get(i));
                    file.write(buf);
                }

                for(i = 0; i < l.n; i++) {
                    byte[] buf = Util.toByteArray(l.rollingMean.get(i));
                    file.write(buf);
                }

                for(i = 0; i < l.n; i++) {
                    byte[] buf = Util.toByteArray(l.rollingVariance.get(i));
                    file.write(buf);
                }
            }
            for(i = 0; i < l.n; ++i){
                float mean = l.binaryWeights.get(i*size);
                if(mean < 0) {
                    mean = -mean;
                }

                byte[] bufMean = Util.toByteArray(mean);
                file.write(bufMean);

                for(j = 0; j < size/8; ++j){
                    int index = i*size + j*8;
                    byte c = 0;
                    for(k = 0; k < 8; ++k){
                        if (j*8 + k >= size) {
                            break;
                        }
                        if (l.binaryWeights.get(index + k) > 0) {
                            c = (byte) (c | 1 << k);
                        }
                    }
                    file.write(c);
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void saveConvolutionalWeights(Layer l, BufferedOutputStream file) {

        if(l.binary != 0){
            //save_convolutional_weights_binary(l, stream);
            //return;
        }

        byte[] buf;

        try {
            int num = l.nweights;

            for(int i = 0; i < l.n; i++) {
                buf = Util.toByteArray(l.biases.get(i));
                file.write(buf);
            }

            if (l.batchNormalize != 0){

                for(int i = 0; i < l.n; i++) {
                    buf = Util.toByteArray(l.scales.get(i));
                    file.write(buf);
                }

                for(int i = 0; i < l.n; i++) {
                    buf = Util.toByteArray(l.rollingMean.get(i));
                    file.write(buf);
                }

                for(int i = 0; i < l.n; i++) {
                    buf = Util.toByteArray(l.rollingVariance.get(i));
                    file.write(buf);
                }
            }

            for(int i = 0; i < num; i++) {
                buf = Util.toByteArray(l.weights.get(i));
                file.write(buf);
            }
        }

        catch (Exception e){
            e.printStackTrace();
        }


    }

    public static void saveBatchnormWeights(Layer l, BufferedOutputStream file) {

        byte[] buf;

        try {

            for(int i = 0; i < l.c; i++) {
                buf = Util.toByteArray(l.scales.get(i));
                file.write(buf);
            }

            for(int i = 0; i < l.c; i++) {
                buf = Util.toByteArray(l.rollingMean.get(i));
                file.write(buf);
            }

            for(int i = 0; i < l.c; i++) {
                buf = Util.toByteArray(l.rollingVariance.get(i));
                file.write(buf);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }


    }

    public static void saveConnectedWeights(Layer l, BufferedOutputStream file) {

        byte[] buf;

        try {

            for(int i = 0; i < l.outputs; i++) {
                buf = Util.toByteArray(l.biases.get(i));
                file.write(buf);
            }

            for(int i = 0; i < l.outputs*l.inputs; i++) {
                buf = Util.toByteArray(l.weights.get(i));
                file.write(buf);
            }

            if(l.batchNormalize != 0) {

                for(int i = 0; i < l.outputs; i++) {
                    buf = Util.toByteArray(l.scales.get(i));
                    file.write(buf);
                }

                for(int i = 0; i < l.outputs; i++) {
                    buf = Util.toByteArray(l.rollingMean.get(i));
                    file.write(buf);
                }

                for(int i = 0; i < l.outputs; i++) {
                    buf = Util.toByteArray(l.rollingVariance.get(i));
                    file.write(buf);
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void saveWeightsUpto(Network net, String filename, int cutoff) {

        byte[] buf;

        try {
            System.out.print(String.format("Saving weights to %s\n",filename));

            FileOutputStream inputStream = new FileOutputStream(new File(filename),true);
            BufferedOutputStream stream = new BufferedOutputStream(inputStream);

            int major = 0;
            int minor = 2;
            int revision = 0;

            buf = Util.toByteArray(major);
            stream.write(buf);

            buf = Util.toByteArray(minor);
            stream.write(buf);

            buf = Util.toByteArray(revision);
            stream.write(buf);

            buf = Util.toByteArray(net.seen.get(0));
            stream.write(buf);

            int i;
            for(i = 0; i < net.seen.get(0) && i < cutoff; ++i){
                                
                Layer l = net.layers[i];
                
                
                if (l.dontsave != 0) {
                    continue;
                }
                if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
                    saveConvolutionalWeights(l, stream);
                } 
                if(l.type == CONNECTED){
                    saveConnectedWeights(l, stream);
                } 
                if(l.type == BATCHNORM){
                    saveBatchnormWeights(l, stream);
                } 
                if(l.type == RNN){
                    saveConnectedWeights(l.inputLayer, stream);
                    saveConnectedWeights(l.selfLayer, stream);
                    saveConnectedWeights(l.outputLayer, stream);
                } 
                if (l.type == LSTM) {
                    saveConnectedWeights(l.wi, stream);
                    saveConnectedWeights(l.wf, stream);
                    saveConnectedWeights(l.wo, stream);
                    saveConnectedWeights(l.wg, stream);
                    saveConnectedWeights(l.ui, stream);
                    saveConnectedWeights(l.uf, stream);
                    saveConnectedWeights(l.uo, stream);
                    saveConnectedWeights(l.ug, stream);
                } 
                if (l.type == GRU) {
                    if(1 != 0){
                        saveConnectedWeights(l.wz, stream);
                        saveConnectedWeights(l.wr, stream);
                        saveConnectedWeights(l.wh, stream);
                        saveConnectedWeights(l.uz, stream);
                        saveConnectedWeights(l.ur, stream);
                        saveConnectedWeights(l.uh, stream);
                    }
                    else{
                        saveConnectedWeights(l.resetLayer, stream);
                        saveConnectedWeights(l.updateLayer, stream);
                        saveConnectedWeights(l.stateLayer, stream);
                    }
                }
                if(l.type == CRNN){
                    saveConvolutionalWeights(l.inputLayer, stream);
                    saveConvolutionalWeights(l.selfLayer, stream);
                    saveConvolutionalWeights(l.outputLayer, stream);
                } 
                if(l.type == LOCAL){

                    int locations = l.outH*l.outW;
                    int size = l.size*l.size*l.c*l.n*locations;

                    for(i = 0; i < l.outputs; i++) {
                        buf = Util.toByteArray(l.biases.get(i));
                        stream.write(buf);
                    }

                    for(i = 0; i < size; i++) {
                        buf = Util.toByteArray(l.weights.get(i));
                        stream.write(buf);
                    }
                }
            }
            stream.close();
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public static void saveWeights(Network net, String filename) {

        saveWeightsUpto(net, filename, net.n);
    }

    public static void transposeMatrix(FloatBuffer a, int rows, int cols) {

        float[] transpose = new float[rows*cols];

        int x, y;
        for(x = 0; x < rows; ++x){
            for(y = 0; y < cols; ++y){
                transpose[y*rows + x] = a.get(x*cols + y);
            }
        }

        Buffers.copy(new FloatBuffer(transpose),a,rows*cols);
    }

    public static void loadConnectedWeights(Layer l, int transpose) {

        try {

            float v;

            for(int i = 0; i < l.outputs; i++) {
                v = GlobalVars.getFloatWeight();
                l.biases.put(i,v);

                //GlobalVars.logStream.write(String.format("%f\n",v));
            }

            for(int i = 0; i < l.outputs*l.inputs; i++) {
                v = GlobalVars.getFloatWeight();
                l.weights.put(i,v);
                //GlobalVars.logStream.write(String.format("%f\n",v));
            }

            if(transpose != 0){
                transposeMatrix(l.weights, l.inputs, l.outputs);
            }

            if (l.batchNormalize != 0 && l.dontloadscales == 0){


                for(int i = 0; i < l.outputs; i++) {
                    v = GlobalVars.getFloatWeight();
                    l.scales.put(i,v);
                    //GlobalVars.logStream.write(String.format("%f\n",v));
                }

                for(int i = 0; i < l.outputs; i++) {
                    v = GlobalVars.getFloatWeight();
                    l.rollingMean.put(i,v);
                    //GlobalVars.logStream.write(String.format("%f\n",v));
                }

                for(int i = 0; i < l.outputs; i++) {
                    v = GlobalVars.getFloatWeight();
                    l.rollingVariance.put(i,v);
                    //GlobalVars.logStream.write(String.format("%f\n",v));
                }
            }
        }

        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void loadBatchnormWeights(Layer l) {

        float v;

        try {
            for(int i = 0; i < l.c; i++) {
                v = GlobalVars.getFloatWeight();
                l.scales.put(i,v);
                //GlobalVars.logStream.write(String.format("%f\n",v));
            }

            for(int i = 0; i < l.c; i++) {
                v = GlobalVars.getFloatWeight();
                l.rollingMean.put(i,v);
                //GlobalVars.logStream.write(String.format("%f\n",v));
            }

            for(int i = 0; i < l.c; i++) {
                v = GlobalVars.getFloatWeight();
                l.rollingVariance.put(i,v);
                //GlobalVars.logStream.write(String.format("%f\n",v));
            }

        }

        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void loadConvolutionalWeightsBinary(Layer l, BufferedInputStream stream) {

        byte[] value;

        try {

            for(int i = 0; i < l.n; i++) {
                value = stream.readNBytes(4);
                l.biases.put(i,Util.byteToFloat(value));
            }

            if(l.batchNormalize != 0 && l.dontloadscales == 0) {
                for(int i = 0; i < l.n; i++) {
                    value = stream.readNBytes(4);
                    l.scales.put(i,Util.byteToFloat(value));
                }

                for(int i = 0; i < l.n; i++) {
                    value = stream.readNBytes(4);
                    l.rollingMean.put(i,Util.byteToFloat(value));
                }

                for(int i = 0; i < l.n; i++) {
                    value = stream.readNBytes(4);
                    l.rollingVariance.put(i,Util.byteToFloat(value));
                }
            }

            int size = l.c*l.size*l.size;
            int i, j, k;
            for(i = 0; i < l.n; ++i){
                float mean = 0;

                value = stream.readNBytes(4);
                mean = Util.byteToFloat(value);

                for(j = 0; j < size/8; ++j){
                    int index = i*size + j*8;
                    byte c = 0;

                    value = stream.readNBytes(1);
                    c = value[0];

                    for(k = 0; k < 8; ++k){
                        if (j*8 + k >= size) {
                            break;
                        }

                        l.weights.put(index + k,((c & 1<<k) != 0) ? mean : -mean);
                    }
                }
            }
        }

        catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void loadConvolutionalWeights(Layer l) {

        if(l.binary != 0){
            //loadConvolutionalWeightsBinary(l, );
            //return;
        }

        if(l.numload != 0) {
            l.n = l.numload;
        }

        int num = l.c/l.groups*l.n*l.size*l.size;

        try {
            for(int i = 0; i < l.n; i++) {
                float v = GlobalVars.getFloatWeight();
                l.biases.put(i,v);
                //GlobalVars.logStream.write(String.format("%f\n",v));
            }

            if (l.batchNormalize != 0 && l.dontloadscales == 0){

                for(int i = 0; i < l.n; i++) {
                    float v = GlobalVars.getFloatWeight();
                    l.scales.put(i,v);
                    //GlobalVars.logStream.write(String.format("%f\n",v));
                }

                for(int i = 0; i < l.n; i++) {
                    float v = GlobalVars.getFloatWeight();
                    l.rollingMean.put(i,v);
                    //GlobalVars.logStream.write(String.format("%f\n",v));
                }

                for(int i = 0; i < l.n; i++) {
                    float v = GlobalVars.getFloatWeight();
                    l.rollingVariance.put(i,v);
                    //GlobalVars.logStream.write(String.format("%f\n",v));
                }
            }

            for(int i = 0; i < num; i++) {
                float v = GlobalVars.getFloatWeight();
                l.weights.put(i,v);
                //GlobalVars.logStream.write(String.format("%f\n",v));
            }

            if (l.flipped != 0) {
                transposeMatrix(l.weights, l.c*l.size*l.size, l.n);
            }
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    public static void loadWeightsUpto(Network net, String filename, int start, int cutoff) {

        try {

            GlobalVars.loadWeights(filename);

//            FileInputStream inputStream = new FileInputStream(new File(filename));
//            BufferedInputStream stream = new BufferedInputStream(inputStream);

            int major;
            int minor;
            int revision;

            major = GlobalVars.getIntWeight();
            //GlobalVars.logStream.write(String.format("%d\n",major));

            minor = GlobalVars.getIntWeight();
            //GlobalVars.logStream.write(String.format("%d\n",minor));

            revision = GlobalVars.getIntWeight();
            //GlobalVars.logStream.write(String.format("%d\n",revision));

            if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){

                long v = GlobalVars.getLongWeight();
                net.seen.put(0, v);
                //GlobalVars.logStream.write(String.format("%d\n",v));
            }
            else {
                int v = GlobalVars.getIntWeight();
                net.seen.put(0,v);
                //GlobalVars.logStream.write(String.format("%d\n",v));
            }

            int transpose = ((major > 1000) || (minor > 1000)) ? 1 : 0;

            int i;

            for(i = start; i < net.n && i < cutoff; ++i){

                Layer l = net.layers[i];

                if (l.dontload != 0) {
                    continue;
                }
                if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
                    loadConvolutionalWeights(l);
                }
                if(l.type == CONNECTED){
                    loadConnectedWeights(l,transpose);
                }
                if(l.type == BATCHNORM){
                    loadBatchnormWeights(l);
                }
                if(l.type == CRNN){
                    loadConvolutionalWeights(l.inputLayer);
                    loadConvolutionalWeights(l.selfLayer);
                    loadConvolutionalWeights(l.outputLayer);
                }
                if(l.type == RNN){
                    loadConnectedWeights(l.inputLayer,transpose);
                    loadConnectedWeights(l.selfLayer,transpose);
                    loadConnectedWeights(l.outputLayer,transpose);
                }
                if (l.type == LSTM) {
                    loadConnectedWeights(l.wi,transpose);
                    loadConnectedWeights(l.wf,transpose);
                    loadConnectedWeights(l.wo,transpose);
                    loadConnectedWeights(l.wg,transpose);
                    loadConnectedWeights(l.ui,transpose);
                    loadConnectedWeights(l.uf,transpose);
                    loadConnectedWeights(l.uo,transpose);
                    loadConnectedWeights(l.ug,transpose);
                }
                if (l.type == GRU) {
                    loadConnectedWeights(l.wz,transpose);
                    loadConnectedWeights(l.wr,transpose);
                    loadConnectedWeights(l.wh,transpose);
                    loadConnectedWeights(l.uz,transpose);
                    loadConnectedWeights(l.ur,transpose);
                    loadConnectedWeights(l.uh,transpose);
                }
                if(l.type == LOCAL){
                    int locations = l.outW*l.outH;
                    int size = l.size*l.size*l.c*l.n*locations;

                    for(int z = 0; z < l.outputs; z++) {
                        float v = GlobalVars.getFloatWeight();
                        l.biases.put(z,v);
                        //GlobalVars.logStream.write(String.format("%f\n",v));
                    }

                    for(int z = 0; z < size; z++) {
                        float v = GlobalVars.getFloatWeight();
                        l.weights.put(z,v);
                        //GlobalVars.logStream.write(String.format("%f\n",v));
                    }
                }
            }
            GlobalVars.freeWeights();
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }        
    }

    public static void loadWeights(Network net, String filename) {

        //GlobalVars.setupLog("C:/darknetaws/logWeightJava.txt");
        
        loadWeightsUpto(net, filename, 0, net.n);
        
//        try {
//            GlobalVars.logStream.close();
//        }
//        catch (Exception e) {
//            e.printStackTrace();
//        }
    }

}
