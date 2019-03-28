package Classes;

import Enums.Activation;
import Enums.CostType;
import Enums.LayerType;

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
    public float learning_rate_scale;
    public float clip;
    public int noloss;
    public int softmax;
    public int classes;
    public int coords;
    public int background;
    public int rescore;
    public int objectness;
    public int joint;
    public int noadjust;
    public int reorg;
    public int log;
    public int tanh;
    public int[] mask;
    public int total;

    public float alpha;
    public float beta;
    public float kappa;

    public float coord_scale;
    public float object_scale;
    public float noobject_scale;
    public float mask_scale;
    public float class_scale;
    public int bias_match;
    public int random;
    public float ignore_thresh;
    public float truth_thresh;
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

    public String cweights;
    public int[] indexes;
    public int[] input_layers;
    public int[] input_sizes;
    public int[] map;
    public int[] counts;
    public float[][] sums;
    public float[] rand;
    public float[] cost;
    public float[] state;
    public float[] prev_state;
    public float[] forgot_state;
    public float[] forgot_delta;
    public float[] state_delta;
    public float[] combine_cpu;
    public float[] combine_delta_cpu;

    public float[] concat;
    public float[] concat_delta;

    public float[] binary_weights;

    public float[] biases;
    public float[] bias_updates;

    public float[] scales;
    public float[] scale_updates;

    public float[] weights;
    public float[] weight_updates;

    public float[] delta;
    public float[] output;
    public float[] loss;
    public float[] squared;
    public float[] norms;

    public float[] spatial_mean;
    public float[] mean;
    public float[] variance;

    public float[] mean_delta;
    public float[] variance_delta;

    public float[] rolling_mean;
    public float[] rolling_variance;

    public float[] x;
    public float[] xNorm;

    public float[] m;
    public float[] v;

    public float[] biasM;
    public float[] biasV;
    public float[] scaleM;
    public float[] scaleV;


    public float[] z_cpu;
    public float[] r_cpu;
    public float[] h_cpu;
    public float[] prev_state_cpu;

    public float[] tempCpu;
    public float[] temp2Cpu;
    public float[] temp3Cpu;

    public float[] dhCpu;
    public float[] hhCpu;
    public float[] prevCellCpu;
    public float[] cellCpu;
    public float[] fCpu;
    public float[] iCpu;
    public float[] gCpu;
    public float[] oCpu;
    public float[] cCpu;
    public float[] dcCpu;

    public float[] binaryInput;

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

}
