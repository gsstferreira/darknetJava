package Classes;

import Enums.Activation;
import Enums.CostType;
import Enums.LayerType;

import java.lang.reflect.Type;
import java.nio.*;

public class Layer {

    public LayerType type;
    public Activation activation;
    public CostType costType;
    
    public int batchNormalize;
    public int shortcut;
    public int batch;
    public int forced;
    public int flipped;
    public int inputs;
    public int outputs;
    public int nweights;
    public int nbiases;
    public int extra;
    public int truths;
    public int h;
    public int w;
    public int c;
    public int outH;
    public int outW;
    public int outC;
    public int n;
    public int maxBoxes;
    public int groups;
    public int size;
    public int side;
    public int stride;
    public int reverse;
    public int flatten;
    public int spatial;
    public int pad;
    public int sqrt;
    public int flip;
    public int index;
    public int binary;
    public int xnor;
    public int steps;
    public int hidden;
    public int truth;
    public float smooth;
    public float dot;
    public float angle;
    public float jitter;
    public float saturation;
    public float exposure;
    public float shift;
    public float ratio;
    public float learningRateScale;
    public float clip;
    public int noloss;
    public int softmax;
    public int classes;
    public int coords;
    public int background;
    public int rescore;
    public int objectness;
    public int joint;
    public int noAdjust;
    public int reorg;
    public int log;
    public int tanh;
    public IntBuffer mask;
    public int total;

    public float alpha;
    public float beta;
    public float kappa;

    public float coordScale;
    public float objectScale;
    public float noobjectScale;
    public float maskScale;
    public float classScale;
    public int biasMatch;
    public int random;
    public float ignoreThresh;
    public float truthThresh;
    public float thresh;
    public float focus;
    public int classfix;
    public int absolute;

    public int onlyforward;
    public int stopbackward;
    public int dontload;
    public int dontsave;
    public int dontloadscales;
    public int numload;

    public float temperature;
    public float probability;
    public float scale;

    public CharBuffer cweights;
    public IntBuffer indexes;
    public IntBuffer inputLayers;
    public IntBuffer inputSizes;
    public IntBuffer map;
    public IntBuffer counts;
    public float[][] sums;
    public FloatBuffer rand;
    public FloatBuffer cost;
    public FloatBuffer state;
    public FloatBuffer prevState;
    public FloatBuffer forgotState;
    public FloatBuffer forgotDelta;
    public FloatBuffer stateDelta;
    public FloatBuffer combineCpu;
    public FloatBuffer combineDeltaCpu;

    public FloatBuffer concat;
    public FloatBuffer concatDelta;

    public FloatBuffer binaryWeights;

    public FloatBuffer biases;
    public FloatBuffer biasUpdates;

    public FloatBuffer scales;
    public FloatBuffer scaleUpdates;

    public FloatBuffer weights;
    public FloatBuffer weightUpdates;

    public FloatBuffer delta;
    public FloatBuffer output;
    public FloatBuffer loss;
    public FloatBuffer squared;
    public FloatBuffer norms;

    public FloatBuffer spatialMean;
    public FloatBuffer mean;
    public FloatBuffer variance;

    public FloatBuffer meanDelta;
    public FloatBuffer varianceDelta;

    public FloatBuffer rollingMean;
    public FloatBuffer rollingVariance;

    public FloatBuffer x;
    public FloatBuffer xNorm;

    public FloatBuffer m;
    public FloatBuffer v;

    public FloatBuffer biasM;
    public FloatBuffer biasV;
    public FloatBuffer scaleM;
    public FloatBuffer scaleV;


    public FloatBuffer zCpu;
    public FloatBuffer rCpu;
    public FloatBuffer hCpu;
    public FloatBuffer prevStateCpu;

    public FloatBuffer tempCpu;
    public FloatBuffer temp2Cpu;
    public FloatBuffer temp3Cpu;

    public FloatBuffer dhCpu;
    public FloatBuffer hhCpu;
    public FloatBuffer prevCellCpu;
    public FloatBuffer cellCpu;
    public FloatBuffer fCpu;
    public FloatBuffer iCpu;
    public FloatBuffer gCpu;
    public FloatBuffer oCpu;
    public FloatBuffer cCpu;
    public FloatBuffer dcCpu;

    public FloatBuffer binaryInput;

    public Layer inputLayer;
    public Layer selfLayer;
    public Layer outputLayer;

    public Layer resetLayer;
    public Layer updateLayer;
    public Layer stateLayer;

    public Layer inputGateLayer;
    public Layer stateGateLayer;
    public Layer inputSaveLayer;
    public Layer stateSaveLayer;
    public Layer inputStateLayer;
    public Layer stateStateLayer;

    public Layer inputZLayer;
    public Layer stateZLayer;

    public Layer inputRLayer;
    public Layer stateRLayer;

    public Layer inputHLayer;
    public Layer stateHLayer;

    public Layer wz;
    public Layer uz;
    public Layer wr;
    public Layer ur;
    public Layer wh;
    public Layer uh;
    public Layer uo;
    public Layer wo;
    public Layer uf;
    public Layer wf;
    public Layer ui;
    public Layer wi;
    public Layer ug;
    public Layer wg;

    public Tree softmaxTree;

    public int workspaceSize;

    public Layer asLayer() {

        return (Layer) this;
    }
}
