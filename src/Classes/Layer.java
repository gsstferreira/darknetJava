package Classes;

import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Yolo.Enums.Activation;
import Yolo.Enums.CostType;
import Yolo.Enums.LayerType;
import Yolo.Layers.*;

import java.nio.ByteBuffer;


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

    public ByteBuffer cweights;
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
