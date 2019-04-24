package Classes;

import Classes.Arrays.ByteArray;
import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Yolo.Enums.Activation;
import Yolo.Enums.CostType;
import Yolo.Enums.LayerType;
import Yolo.Layers.*;

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
    public int nWeights;
    public int nBiases;
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
    public int noLoss;
    public int softmax;
    public int classes;
    public int coords;
    public int background;
    public int rescore;
//    public int objectness;
//    public int joint;
    public int noAdjust;
    public int reorg;
    public int log;
    public int tanh;
    public IntArray mask;
    public int total;

    public float alpha;
    public float beta;
    public float kappa;

    public float coordScale;
    public float objectScale;
    public float noObjectScale;
    public float maskScale;
    public float classScale;
    public int biasMatch;
    public int random;
    public float ignoreThresh;
    public float truthThresh;
    public float thresh;
//    public float focus;
    public int classFix;
    public int absolute;

    public int onlyForward;
    public int stopBackward;
    public int dontLoad;
    public int dontSave;
    public int dontLoadScales;
    public int numLoad;

    public float temperature;
    public float probability;
    public float scale;

    public ByteArray cWeights;
    public IntArray indexes;
    public IntArray inputLayers;
    public IntArray inputSizes;
    public IntArray map;
    public IntArray counts;
    public float[][] sums;
    public FloatArray rand;
    public FloatArray cost;
    public FloatArray state;
    public FloatArray prevState;
    public FloatArray forgotState;
    public FloatArray forgotDelta;
//    public FloatArray stateDelta;
//    public FloatArray combineCpu;
//    public FloatArray combineDeltaCpu;
//
//    public FloatArray concat;
//    public FloatArray concatDelta;

    public FloatArray binaryWeights;

    public FloatArray biases;
    public FloatArray biasUpdates;

    public FloatArray scales;
    public FloatArray scaleUpdates;

    public FloatArray weights;
    public FloatArray weightUpdates;

    public FloatArray delta;
    public FloatArray output;
    public FloatArray loss;
    public FloatArray squared;
    public FloatArray norms;

//    public FloatArray spatialMean;
    public FloatArray mean;
    public FloatArray variance;

    public FloatArray meanDelta;
    public FloatArray varianceDelta;

    public FloatArray rollingMean;
    public FloatArray rollingVariance;

    public FloatArray x;
    public FloatArray xNorm;

    public FloatArray m;
    public FloatArray v;

    public FloatArray biasM;
    public FloatArray biasV;
    public FloatArray scaleM;
    public FloatArray scaleV;


    public FloatArray zCpu;
    public FloatArray rCpu;
    public FloatArray hCpu;
    public FloatArray prevStateCpu;

    public FloatArray tempCpu;
    public FloatArray temp2Cpu;
    public FloatArray temp3Cpu;

    public FloatArray dhCpu;
//    public FloatArray hhCpu;
    public FloatArray prevCellCpu;
    public FloatArray cellCpu;
    public FloatArray fCpu;
    public FloatArray iCpu;
    public FloatArray gCpu;
    public FloatArray oCpu;
    public FloatArray cCpu;
    public FloatArray dcCpu;

    public FloatArray binaryInput;

    public Layer inputLayer;
    public Layer selfLayer;
    public Layer outputLayer;

//    public Layer resetLayer;
//    public Layer updateLayer;
//    public Layer stateLayer;
//
//    public Layer inputGateLayer;
//    public Layer stateGateLayer;
//    public Layer inputSaveLayer;
//    public Layer stateSaveLayer;
//    public Layer inputStateLayer;
//    public Layer stateStateLayer;
//
//    public Layer inputZLayer;
//    public Layer stateZLayer;
//
//    public Layer inputRLayer;
//    public Layer stateRLayer;
//
//    public Layer inputHLayer;
//    public Layer stateHLayer;

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

    public long workspaceSize;
    
    public void forwardLayer(Network net) {
        
        switch (this.type) {
            case CONVOLUTIONAL:
                ((ConvolutionalLayer)this).forward(net);
                break;
            case DECONVOLUTIONAL:
                ((DeconvolutionalLayer)this).forward(net);
                break;
            case LOCAL:
                ((LocalLayer)this).forward(net);
                break;
            case ACTIVE:
                ((ActivationLayer)this).forward(net);
                break;
            case LOGXENT:
                ((LogisticLayer)this).forward(net);
                break;
            case L2NORM:
                ((L2NormLayer)this).forward(net);
                break;
            case RNN:
                ((RnnLayer)this).forward(net);
                break;
            case GRU:
                ((GruLayer)this).forward(net);
                break;
            case LSTM:
                ((LstmLayer)this).forward(net);
                break;
            case CRNN:
                ((CrnnLayer)this).forward(net);
                break;
            case CONNECTED:
                ((ConnectedLayer)this).forward(net);
                break;
            case CROP:
                ((CropLayer)this).forward(net);
                break;
            case COST:
                ((CostLayer)this).forward(net);
                break;
            case REGION:
                ((RegionLayer)this).forward(net);
                break;
            case YOLO:
                ((YoloLayer)this).forward(net);
                break;
            case ISEG:
                ((IsegLayer)this).forward(net);
                break;
            case DETECTION:
                ((DetectionLayer)this).forward(net);
                break;
            case SOFTMAX:
                ((SoftmaxLayer)this).forward(net);
                break;
            case NORMALIZATION:
                ((NormalizationLayer)this).forward(net);
                break;
            case BATCHNORM:
                ((BatchnormLayer)this).forward(net);
                break;
            case MAXPOOL:
                ((MaxpoolLayer)this).forward(net);
                break;
            case REORG:
                ((ReorgLayer)this).forward(net);
                break;
            case AVGPOOL:
                ((AvgPoolLayer)this).forward(net);
                break;
            case ROUTE:
                ((RouteLayer)this).forward(net);
                break;
            case UPSAMPLE:
                ((UpsampleLayer)this).forward(net);
                break;
            case SHORTCUT:
                ((ShortcutLayer)this).forward(net);
                break;
            case DROPOUT:
                ((DropoutLayer)this).forward(net);
                break;
            default:
                System.out.println(String.format("Layer::forward - Layer type not recognized: '%s'\n", type.name()));
                break;
        }
    }
}
