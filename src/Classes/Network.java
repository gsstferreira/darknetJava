package Classes;

import Classes.Arrays.DetectionArray;
import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Classes.Arrays.LongArray;
import Yolo.Enums.LayerType;
import Yolo.Enums.LearningRatePolicy;
import Yolo.Layers.DetectionLayer;
import Yolo.Layers.RegionLayer;
import Yolo.Layers.YoloLayer;
import Yolo.Parser;

public class Network implements Cloneable {

    @Override
    public Network clone() throws CloneNotSupportedException {
        return (Network) super.clone();
    }

    public Network tryClone() {

        try {
            return this.clone();
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
            return null;
        }
    }

    public int n;
    public int batch;

    public LongArray seen;
    public IntArray t;
//    public float epoch;
    public int subdivisions;
    public Layer[] layers;
    public FloatArray output;

    public LearningRatePolicy policy;

    public float learningRate;
    public float momentum;
    public float decay;
    public float gamma;
    public float scale;
    public float power;
    public int timeSteps;
    public int step;
    public int maxBatches;
    public FloatArray scales;
    public IntArray steps;
    public int numSteps;
    public int burnIn;

    public int adam;
    public float B1;
    public float B2;
    public float eps;

    public int inputs;
    public int outputs;
    public int truths;
    public int noTruth;
    public int h;
    public int w;
    public int c;
    public int maxCrop;
    public int minCrop;
    public float maxRatio;
    public float minRatio;
    public int center;
    public float angle;
    public float aspect;
    public float exposure;
    public float saturation;
    public float hue;
    public int random;

    public int gpuIndex;
    public Tree hierarchy;

    public FloatArray input;
    public FloatArray truth;
    public FloatArray delta;
    public FloatArray workspace;
    public int train;
    public int index;
    public FloatArray cost;
    public float clip;

    public Network(){}

    public Network(int n) {

        this.n = n;
        this.layers = new Layer[n];
        this.seen = new LongArray(1);
        this.t    = new IntArray(1);
        this.cost = new FloatArray(1);
    }

    public static Network loadNetwork(String cfgFile, String weightsFile, int clear) {

        Network net = Parser.parseNetworkCfg(cfgFile);

        if(weightsFile != null && !weightsFile.equals("")) {
            Parser.loadWeights(net,weightsFile);
        }

        if(clear != 0) {
            net.seen.set(0,0);
        }
        return net;
    }

//    public LoadArgs getbaseArgs() {
//
//        LoadArgs args = new LoadArgs();
//        args.w = this.w;
//        args.h = this.h;
//        args.size = this.w;
//
//        args.min = this.minCrop;
//        args.max = this.maxCrop;
//        args.angle = this.angle;
//        args.aspect = this.aspect;
//        args.exposure = this.exposure;
//        args.center = this.center;
//        args.saturation = this.saturation;
//        args.hue = this.hue;
//
//        return args;
//    }
    
    public Layer getNetworkOutputLayer() {

        int i;
        for(i = this.n - 1; i >= 0; --i){
            if(this.layers[i].type != LayerType.COST) {
                break;
            }
        }
        return this.layers[i];
    }

    public void setBatchNetwork(int b) {

        this.batch = b;

        for(int i = 0; i < this.n; ++i){
            this.layers[i].batch = b;
        }
    }

    public Detection[] getBoxes(int w, int h, float thresh, float hier, IntArray map, int relative, IntArray num) {

        Detection[] dets = makeBoxes(thresh, num);
        fillBoxes(w, h, thresh, hier, map, relative, dets);
        return dets;
    }

//    FloatArray predictImage(Image im) {
//
//        Image imr = im.letterbox(this.w, this.h);
//        this.setBatchNetwork(1);
//        return this.predict(imr.data);
//    }

    @SuppressWarnings("UnusedReturnValue")
    public FloatArray predict(FloatArray input) {

        Network clone = this.tryClone();

        clone.input = input;
        clone.truth = null;
        clone.train = 0;
        clone.delta = null;

        clone.forward();

        return clone.output;
    }

    public Detection[] makeBoxes(float thresh, IntArray num) {

        Layer l = layers[n - 1];
        int i;
        int nboxes = numDetections(thresh);

        if(num != null) {
            num.set(0,nboxes);
        }

        Detection[] dets = new Detection[nboxes];

        for(i = 0; i < nboxes; ++i){

            dets[i] = new Detection();
            dets[i].prob = new float[l.classes];
            if(l.coords > 4){
                dets[i].mask = new float[l.coords - 4];
            }
        }
        return dets;
    }

    public void fillBoxes(int w, int h, float thresh, float hier, IntArray map, int relative, Detection[] dets) {

        int j;

        DetectionArray detBuffer = new DetectionArray(dets);

        for(j = 0; j < n; ++j){
            Layer l = this.layers[j];

            if(l.type == LayerType.YOLO){

                int count = ((YoloLayer)l).getYoloDetections(w, h, this.w, this.h, thresh, map, relative, detBuffer);
                detBuffer.offset(count);
            }

            if(l.type == LayerType.REGION){

                //noinspection ConstantConditions
                ((RegionLayer)l).getRegionDetections( w, h, this.w, this.h, thresh, map, hier, relative, detBuffer);
                detBuffer.offset(l.w*l.h*l.n);
            }

            if(l.type == LayerType.DETECTION){

                //noinspection ConstantConditions
                ((DetectionLayer)l).getDetectionDetections(w, h, thresh, detBuffer);
                detBuffer.offset(l.w*l.h*l.n);
            }
        }
    }

    public int numDetections(float thresh) {

        int i;
        int s = 0;
        for(i = 0; i < n; ++i){
            Layer l = layers[i];
            if(l.type == LayerType.YOLO){
                s += ((YoloLayer)l).numDetections(thresh);
            }
            if(l.type == LayerType.DETECTION || l.type == LayerType.REGION){
                s += l.w*l.h*l.n;
            }
        }
        return s;
    }

    public void forward() {

        int i;
        for(i = 0; i < n; ++i){
            index = i;
            Layer l = layers[i];
            if(l.delta != null){
                l.delta.setAll(0,l.outputs*l.batch);
            }

            l.forwardLayer(this);
            input = l.output;
            if(l.truth != 0) {
                truth = l.output;
            }
        }
        calcCost();
    }

    public void calcCost() {

        int i;
        float sum = 0;
        int count = 0;
        for(i = 0; i < n; ++i){
            if(layers[i].cost != null){
                sum += layers[i].cost.get(0);
                ++count;
            }
        }
        this.cost.set(0,sum/count);
    }
}
