package Classes;

import Enums.LearningRatePolicy;

public class Network {

    public int n;
    public int batch;

    public int[] seen;
    public int[] t;
    public float epoch;
    public int subdivisions;
    public Layer[] layers;
    public float[] output;

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
    public float[] scales;
    public int[] steps;
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
    public Tree[] hierarchy;

    public float[] input;
    public float[] truth;
    public float[] delta;
    public float[] workspace;
    public int train;
    public int index;
    public float[] cost;
    public float clip;

    public float[] inputGpu;
    public float[] truthGpu;
    public float[] deltaGpu;
    public float[] outputGpu;

}
