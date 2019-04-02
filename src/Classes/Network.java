package Classes;

import Enums.LearningRatePolicy;

import java.nio.*;

public class Network {

    public int n;
    public int batch;

    public IntBuffer seen;
    public IntBuffer t;
    public float epoch;
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
    public Tree[] hierarchy;

    public FloatBuffer input;
    public FloatBuffer truth;
    public FloatBuffer delta;
    public FloatBuffer workspace;
    public int train;
    public int index;
    public FloatBuffer cost;
    public float clip;

    public FloatBuffer inputGpu;
    public FloatBuffer truthGpu;
    public FloatBuffer deltaGpu;
    public FloatBuffer outputGpu;

}
