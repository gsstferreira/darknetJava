package Classes;

import Classes.Buffers.DetectionBuffer;
import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Classes.Buffers.LongBuffer;
import Tools.Blas;
import Tools.Buffers;
import Yolo.Enums.LayerType;
import Yolo.Enums.LearningRatePolicy;
import Yolo.Layers.DetectionLayer;
import Yolo.Layers.RegionLayer;
import Yolo.Layers.YoloLayer;
import Yolo.Parser;

public class Network implements Cloneable {

    @Override
    public Object clone() throws CloneNotSupportedException {
        Network n = (Network) super.clone();

        if(n.workspace != null){n.workspace = Buffers.copyNew(n.workspace,n.workspace.size());}
        if(n.seen != null){n.seen = Buffers.copyNew(n.seen,n.seen.size());}
        if(n.t != null){n.t = Buffers.copyNew(n.t,n.t.size());}
        if(n.output != null){n.output = Buffers.copyNew(n.output,n.output.size());}
        if(n.cost != null){n.cost = Buffers.copyNew(n.cost,n.cost.size());}

        if(n.scales != null){n.scales = Buffers.copyNew(n.scales,n.scales.size());}
        if(n.steps != null){n.steps = Buffers.copyNew(n.steps,n.steps.size());}

        if(n.input != null){n.input = Buffers.copyNew(n.input,n.input.size());}
        if(n.truth != null){n.truth = Buffers.copyNew(n.truth,n.truth.size());}
        if(n.delta != null){n.delta = Buffers.copyNew(n.delta,n.delta.size());}

        return n;
    }

    public Network tryClone() {

        try {
            return (Network) this.clone();
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
            return null;
        }
    }

    public int n;
    public int batch;

    public LongBuffer seen;
    public IntBuffer t;
//    public float epoch;
    public int subdivisions;
    public Layer[] layers;
    public FloatBuffer output;

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
    public FloatBuffer scales;
    public IntBuffer steps;
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

    public FloatBuffer input;
    public FloatBuffer truth;
    public FloatBuffer delta;
    public FloatBuffer workspace;
    public int train;
    public int index;
    public FloatBuffer cost;
    public float clip;

    public Network(){}

    public Network(int n) {

        this.n = n;
        this.layers = new Layer[n];
        this.seen = new LongBuffer(1);
        this.t    = new IntBuffer(1);
        this.cost = new FloatBuffer(1);
    }

    public static Network loadNetwork(String cfgFile, String weightsFile, int clear) {

        Network net = Parser.parseNetworkCfg(cfgFile);

        if(weightsFile != null && !weightsFile.equals("")) {
            Parser.loadWeights(net,weightsFile);
        }

        if(clear != 0) {
            net.seen.put(0,0);
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

    public Detection[] getBoxes(int w, int h, float thresh, float hier, IntBuffer map, int relative, IntBuffer num) {

        Detection[] dets = makeBoxes(thresh, num);
        fillBoxes(w, h, thresh, hier, map, relative, dets);
        return dets;
    }

//    FloatBuffer predictImage(Image im) {
//
//        Image imr = im.letterbox(this.w, this.h);
//        this.setBatchNetwork(1);
//        return this.predict(imr.data);
//    }

    public FloatBuffer predict(FloatBuffer input) {

        Network clone = this.tryClone();

        clone.input = input;
        clone.truth = null;
        clone.train = 0;
        clone.delta = null;

        clone.forward();

        return clone.output;
    }

    public Detection[] makeBoxes(float thresh, IntBuffer num) {

        Layer l = layers[n - 1];
        int i;
        int nboxes = numDetections(thresh);

        if(num != null) {
            num.put(0,nboxes);
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

    public void fillBoxes( int w, int h, float thresh, float hier, IntBuffer map, int relative, Detection[] dets) {

        int j;

        DetectionBuffer detBuffer = new DetectionBuffer(dets);

        for(j = 0; j < n; ++j){
            Layer l = this.layers[j];

            if(l.type == LayerType.YOLO){

                int count = ((YoloLayer)l).getYoloDetections(w, h, this.w, this.h, thresh, map, relative, detBuffer);
                detBuffer.offset(count);
            }

            if(l.type == LayerType.REGION){

                ((RegionLayer)l).getRegionDetections( w, h, this.w, this.h, thresh, map, hier, relative, detBuffer);
                detBuffer.offset(l.w*l.h*l.n);
            }

            if(l.type == LayerType.DETECTION){

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
                Blas.fillCpu(l.outputs * l.batch, 0, l.delta, 1);
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
        this.cost.put(0,sum/count);
    }
}
