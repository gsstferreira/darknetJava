package Layers;

import Classes.Image;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Enums.Activation;
import Enums.LayerType;
import Tools.*;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

public class ConvolutionalLayer extends Layer {

    private void swapBinary() {
        
        FloatBuffer swap = weights;
        weights = binaryWeights;
        binaryWeights = swap;
    }

    private static void binarizeWeights(FloatBuffer weights, int n, int size, FloatBuffer binary) {
        
        int i, f;
        for(f = 0; f < n; ++f){
            float mean = 0;
            for(i = 0; i < size; ++i){
                mean += Math.abs(weights.get(f*size + i));
            }
            mean = mean / size;
            for(i = 0; i < size; ++i){
                
                binary.put(f*size + i,(weights.get(f*size + i) > 0) ? mean : -mean);
            }
        }
    }

    private static void binarizeCpu(FloatBuffer input, int n, FloatBuffer binary) {
        
        int i;
        for(i = 0; i < n; ++i){
            
            binary.put(i,(input.get(i) > 0) ? 1 : -1);
        }
    }

    private void binarizeInput(FloatBuffer input, int n, int size, FloatBuffer binary) {
        int i, s;
        for(s = 0; s < size; ++s){
            float mean = 0;
            for(i = 0; i < n; ++i){
                mean += Math.abs(input.get(i*size + s));
            }
            mean = mean / n;
            for(i = 0; i < n; ++i){
                
                binary.put(i*size + s, (input.get(i*size +s) > 0) ? mean : -mean);
            }
        }
    }

    public int convolutionalOutHeight() {
        
        return (h + 2*pad - size) / stride + 1;
    }

    public int convolutionalOutWidth() {
        
        return (w + 2*pad - size) / stride + 1;
    }

    public Image getConvolutionalImage() {
        
        return new Image(outW,outH,outC,output);
    }

    public Image getConvolutionalDelta() {
        
        return new Image(outW,outH,outC,delta);
    }
    
    public static int getWorkspaceSize(Layer l){

        return l.outH*l.outW*l.size*l.size*l.c/l.groups*Float.SIZE;
    }
    
    public ConvolutionalLayer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding,
                              Activation activation, int batchNormalize, int binary, int xnor, int adam) {
        int i;
        type = LayerType.CONVOLUTIONAL;

        this.groups = groups;
        this.h = h;
        this.w = w;
        this.c = c;
        this.n = n;
        this.binary = binary;
        this.xnor = xnor;
        this.batch = batch;
        this.stride = stride;
        this.size = size;
        this.pad = padding;
        this.batchNormalize = batchNormalize;

        weights = BufferUtils.createFloatBuffer(c/groups*n*size*size);
        weightUpdates = BufferUtils.createFloatBuffer(c/groups*n*size*size);

        biases = BufferUtils.createFloatBuffer(n);
        biasUpdates = BufferUtils.createFloatBuffer(n);

        nweights = c/groups*n*size*size;
        nbiases = n;

        //TODO remove float cast?
        float scale = (float) Math.sqrt(2./(float)(size*size*c/groups));


        for(i = 0; i < nweights; ++i) {
            weights.put(i,scale* Rand.randNormal());
        }

        this.outH = convolutionalOutHeight();
        this.outW = convolutionalOutWidth();
        this.outC = n;
        this.outputs = outH * outW * outC;
        this.inputs = this.w * this.h * this.c;

        output = BufferUtils.createFloatBuffer(this.batch*outputs);
        delta  = BufferUtils.createFloatBuffer(this.batch*outputs);

        if(binary != 0){
            binaryWeights = BufferUtils.createFloatBuffer(this.nweights);
            cweights = BufferUtils.createCharBuffer(this.nweights);
            scales = BufferUtils.createFloatBuffer(n);
        }
        if(xnor != 0){
            binaryWeights = BufferUtils.createFloatBuffer(this.nweights);
            binaryInput = BufferUtils.createFloatBuffer(this.batch*inputs);
        }
        if(batchNormalize != 0){

            scales = BufferUtils.createFloatBuffer(n);
            scaleUpdates = BufferUtils.createFloatBuffer(n);

            for(i = 0; i < n; ++i){
                scales.put(i,1);
            }

            mean = BufferUtils.createFloatBuffer(n);
            variance = BufferUtils.createFloatBuffer(n);

            meanDelta = BufferUtils.createFloatBuffer(n);
            varianceDelta = BufferUtils.createFloatBuffer(n);

            rollingMean = BufferUtils.createFloatBuffer(n);
            rollingVariance = BufferUtils.createFloatBuffer(n);
            x = BufferUtils.createFloatBuffer(this.batch*outputs);
            xNorm = BufferUtils.createFloatBuffer(this.batch*outputs);
        }
        if(adam != 0){
            m = BufferUtils.createFloatBuffer(this.nweights);
            v = BufferUtils.createFloatBuffer(this.nweights);
            biasM = BufferUtils.createFloatBuffer(n);
            scaleM = BufferUtils.createFloatBuffer(n);
            biasV = BufferUtils.createFloatBuffer(n);
            scaleV = BufferUtils.createFloatBuffer(n);
        }

        this.workspaceSize = getWorkspaceSize(this);
        this.activation = activation;
    }

    public void denormalize() {

        int i, j;
        for(i = 0; i < n; ++i){

            float scale = scales.get(i)/(float)Math.sqrt(rollingVariance.get(i) + 0.00001f);

            for(j = 0; j < c/groups*size*size; ++j){

                weights.put(i*c/groups*size*size + j,weights.get(i*c/groups*size*size + j) * scale);
            }

            biases.put(i,biases.get(i) - rollingMean.get(i)*scale);
            scales.put(i,1);
            rollingMean.put(i,0);
            rollingVariance.put(i,1);
        }
    }

    public void resize(int width, int height) {
        
        w = width;
        h = height;

        int out_w = convolutionalOutWidth();
        int out_h = convolutionalOutHeight();

        outW = out_w;
        outH = out_h;

        outputs = outH * outW * outC;
        inputs = w * h * c;

        output = BufferUtil.reallocBuffer(output,batch*outputs);
        delta = BufferUtil.reallocBuffer(delta,batch*outputs);
        
        if(batchNormalize != 0){

            x = BufferUtil.reallocBuffer(x,batch*outputs);
            xNorm = BufferUtil.reallocBuffer(xNorm,batch*outputs);
            
        }
        
        workspaceSize = getWorkspaceSize(this);
    }

    public static void addBias(FloatBuffer output, FloatBuffer biases, int batch, int n, int size) {
        
        int i,j,b;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < n; ++i){
                for(j = 0; j < size; ++j){

                    int index = (b * n + i) * size + j;
                    float val = output.get(index) + biases.get(i);
                    output.put(index,val);
                }
            }
        }
    }

    public static void scaleBias(FloatBuffer output, FloatBuffer scales, int batch, int n, int size) {
        
        int i,j,b;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < n; ++i){
                for(j = 0; j < size; ++j){

                    int index = (b * n + i) * size + j;
                    float val = output.get(index) * scales.get(i);
                    output.put(index,val);
                }
            }
        }
    }

    public static void backwardBias(FloatBuffer bias_updates, FloatBuffer delta, int batch, int n, int size) {
        
        int i,b;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < n; ++i){
                
                FloatBuffer _delta = BufferUtil.offsetBuffer(delta,size*(i+b*n));
                bias_updates.put(i,bias_updates.get(i) + Util.sumArray(_delta, size));
            }
        }
    }

    public void forward(Network net) {

        int i, j;

        Blas.fillCpu(outputs*batch,0,output,1);

        if(xnor != 0){
            binarizeWeights(weights, n, c/groups*size*size, binaryWeights);
            swapBinary();
            binarizeCpu(net.input, c*h*w*batch, binaryInput);
            net.input = binaryInput;
        }

        int m = n/groups;
        int k = size*size*c/groups;
        int n = outW*outH;
        for(i = 0; i < batch; ++i){
            for(j = 0; j < groups; ++j){

                FloatBuffer _a = BufferUtil.offsetBuffer(weights,j*nweights/groups);
                FloatBuffer _b = net.workspace;
                FloatBuffer _c = BufferUtil.offsetBuffer(output,(i*groups + j)*n*m);
                FloatBuffer im = BufferUtil.offsetBuffer(net.input,(i*groups + j)*c/groups*h*w);

                if (size == 1) {
                    _b = im;
                } else {
                    ImCol.im2ColCpu(im, c/groups, h, w, size, stride, pad, _b);
                }
                Gemm.gemm(0,0,m,n,k,1,_a,k,_b,n,1,_c,n);
            }
        }

        if(batchNormalize != 0){
            BatchnormLayer.staticForward(this, net);
        }
        else {
            addBias(output, biases, batch, n, outH*outW);
        }

        Activation.activateArray(output, outputs*batch, activation);
        if(binary != 0 || xnor != 0) {
            swapBinary();
        }
    }

    public void backward(Network net) {

        int i, j;
        int m = n/groups;
        int _n = size*size*c/groups;
        int k = outW*outH;

        Activation.gradientArray(output, outputs*batch, activation, delta);

        if(batchNormalize != 0){
            BatchnormLayer.staticBackward(this, net);
        }
        else {
            backwardBias(biasUpdates, delta, batch, n, k);
        }

        for(i = 0; i < batch; ++i){
            for(j = 0; j < groups; ++j){

                FloatBuffer _a = BufferUtil.offsetBuffer(delta,(i*groups + j)*m*k);
                FloatBuffer _b = net.workspace;
                FloatBuffer _c = BufferUtil.offsetBuffer(weightUpdates,j*nweights/groups);

                FloatBuffer im  = BufferUtil.offsetBuffer(net.input,(i * groups + j)*c/groups*h*w);
                FloatBuffer imd = BufferUtil.offsetBuffer(net.delta,(i * groups + j)*c/groups*h*w);

                if(size == 1){
                    _b = im;
                }
                else {
                    ImCol.im2ColCpu(im, c/groups, h, w, size, stride, pad, _b);
                }

                Gemm.gemm(0,1,m,_n,k,1,_a,k,_b,k,1,_c,_n);

                if (net.delta != null) {

                    _a = BufferUtil.offsetBuffer(weights,j*nweights/groups);
                    _b = BufferUtil.offsetBuffer(delta,(i*groups + j)*m*k);
                    _c = net.workspace;

                    if (size == 1) {
                        _c = imd;
                    }

                    Gemm.gemm(1,0,_n,k,m,1,_a,_n,_b,k,0,_c,k);

                    if (size != 1) {
                        ImCol.col2ImCpu(net.workspace, c/groups, h, w, size, stride, pad, imd);
                    }
                }
            }
        }
    }

    public void update(UpdateArgs a) {

        float learning_rate = a.learningRate*learningRateScale;
        float momentum = a.momentum;
        float decay = a.decay;
        int batch = a.batch;

        Blas.axpyCpu(n, learning_rate/batch, biasUpdates, 1, biases, 1);
        Blas.scalCpu(n, momentum, biasUpdates, 1);

        if(scales != null){
            Blas.axpyCpu(n, learning_rate/batch, scaleUpdates, 1, scales, 1);
            Blas.scalCpu(n, momentum, scaleUpdates, 1);
        }

        Blas.axpyCpu(nweights, -decay*batch, weights, 1, weightUpdates, 1);
        Blas.axpyCpu(nweights, learning_rate/batch, weightUpdates, 1, weights, 1);
        Blas.scalCpu(nweights, momentum, weightUpdates, 1);
    }

    public Image getConvolutionalWeight(int i) {

        int h = size;
        int w = size;
        int _c = c/groups;

        FloatBuffer fb = BufferUtil.offsetBuffer(weights,i*h*w*_c);
        return new Image(w,h,_c,fb);
    }

    public void rgbgrWeights() {

        int i;
        for(i = 0; i < n; ++i){
            Image im = getConvolutionalWeight(i);
            if (im.c == 3) {
                im.rgbgr();
            }
        }
    }

    public void rescaleWeights(float scale, float trans) {

        int i;
        for(i = 0; i < n; ++i){
            Image im = getConvolutionalWeight(i);
            if (im.c == 3) {

                im.scale(scale);
                float sum = Util.sumArray(im.data, im.w*im.h*im.c);
                biases.put(i,biases.get(i) + sum*trans);
            }
        }
    }

    public Image[] getWeights() {

        Image[] imWeights = new Image[n];

        int i;
        for(i = 0; i < n; ++i){

            imWeights[i] = getConvolutionalWeight(i).copyImage();
            imWeights[i].normalize();
        }
        return imWeights;
    }

    public Image[] visualizeConvolutionalLayer(char[] window, Image[] prevWeights)
    {
        return getWeights();
    }
}
