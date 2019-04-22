package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Tools.*;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;
import org.lwjgl.BufferUtils;


public class ConvolutionalLayer extends Layer {

    private void swapBinary() {
        
        FloatBuffer swap = weights;
        weights = binaryWeights;
        binaryWeights = swap;
    }

    public static void binarizeWeights(FloatBuffer weights, int n, int size, FloatBuffer binary) {
        
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

//    private void binarizeInput(FloatBuffer input, int n, int size, FloatBuffer binary) {
//        int i, s;
//        for(s = 0; s < size; ++s){
//            float mean = 0;
//            for(i = 0; i < n; ++i){
//                mean += Math.abs(input.get(i*size + s));
//            }
//            mean = mean / n;
//            for(i = 0; i < n; ++i){
//
//                binary.put(i*size + s, (input.get(i*size +s) > 0) ? mean : -mean);
//            }
//        }
//    }

    public int convolutionalOutHeight() {
        
        return (h + 2*pad - size) / stride + 1;
    }

    public int convolutionalOutWidth() {
        
        return (w + 2*pad - size) / stride + 1;
    }

//    public Image getConvolutionalImage() {
//
//        return new Image(outW,outH,outC,output);
//    }
//
//    public Image getConvolutionalDelta() {
//
//        return new Image(outW,outH,outC,delta);
//    }
    
    private long getWorkspaceSize(){

        return outH*outW*size*size*c/groups;
    }
    
    public ConvolutionalLayer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding,
                              Activation activation, int batchNormalize, int binary, int xnor, int adam) {

        int sscg = size*size*c/groups;
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

        this.nweights = sscg*n;

        this.weights = new FloatBuffer(this.nweights);
        this.weightUpdates = new FloatBuffer(this.nweights);

        this.biases = new FloatBuffer(n);
        this.biasUpdates = new FloatBuffer(n);

        this.nbiases = n;

        float scale = (float) Math.sqrt(2./sscg);

        for(i = 0; i < nweights; ++i) {
            weights.put(i,scale * Rand.randNormal());
        }

        this.outH = convolutionalOutHeight();
        this.outW = convolutionalOutWidth();
        this.outC = n;
        this.outputs = outH * outW * outC;
        this.inputs = this.w * this.h * this.c;

        this.output = new FloatBuffer(this.batch*outputs);
        this.delta  = new FloatBuffer(this.batch*outputs);

        if(binary != 0){
            this.binaryWeights = new FloatBuffer(this.nweights);
            this.cweights = BufferUtils.createByteBuffer(this.nweights);
            this.scales = new FloatBuffer(n);
        }
        if(xnor != 0){
            this.binaryWeights = new FloatBuffer(this.nweights);
            this.binaryInput = new FloatBuffer(this.batch*inputs);
        }
        if(batchNormalize != 0){

            this.scales = new FloatBuffer(n);
            this.scaleUpdates = new FloatBuffer(n);

            for(i = 0; i < n; ++i){
                this.scales.put(i,1);
            }

            this.mean = new FloatBuffer(n);
            this.variance = new FloatBuffer(n);

            this.meanDelta = new FloatBuffer(n);
            this.varianceDelta = new FloatBuffer(n);

            this.rollingMean = new FloatBuffer(n);
            this.rollingVariance = new FloatBuffer(n);
            this.x = new FloatBuffer(this.batch*outputs);
            this.xNorm = new FloatBuffer(this.batch*outputs);
        }
        if(adam != 0){
            this.m = new FloatBuffer(this.nweights);
            this.v = new FloatBuffer(this.nweights);
            this.biasM = new FloatBuffer(n);
            this.scaleM = new FloatBuffer(n);
            this.biasV = new FloatBuffer(n);
            this.scaleV = new FloatBuffer(n);
        }

        this.workspaceSize = getWorkspaceSize();
        this.activation = activation;

        float flops = (this.nweights * this.outH*this.outW)/500000000.0f;
        System.out.printf("Conv %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, outW, outH, outC, flops);
    }

//    public void denormalize() {
//
//        int i, j;
//        for(i = 0; i < n; ++i){
//
//            float scale = scales.get(i)/(float)Math.sqrt(rollingVariance.get(i) + 0.00001f);
//
//            for(j = 0; j < c/groups*size*size; ++j){
//
//                weights.put(i*c/groups*size*size + j,weights.get(i*c/groups*size*size + j) * scale);
//            }
//
//            biases.put(i,biases.get(i) - rollingMean.get(i)*scale);
//            scales.put(i,1);
//            rollingMean.put(i,0);
//            rollingVariance.put(i,1);
//        }
//    }

    public void resize(int width, int height) {
        
        w = width;
        h = height;

        int out_w = convolutionalOutWidth();
        int out_h = convolutionalOutHeight();

        outW = out_w;
        outH = out_h;

        outputs = outH * outW * outC;
        inputs = w * h * c;

        output = Buffers.realloc(output,batch*outputs);
        delta = Buffers.realloc(delta,batch*outputs);
        
        if(batchNormalize != 0){

            x = Buffers.realloc(x,batch*outputs);
            xNorm = Buffers.realloc(xNorm,batch*outputs);
            
        }
        
        workspaceSize = getWorkspaceSize();
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
                
                FloatBuffer _delta = delta.offsetNew(size*(i+b*n));
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

        FloatBuffer _a = weights.shallowClone();
        FloatBuffer _b = net.workspace.shallowClone();
        FloatBuffer _c = output.shallowClone();
        FloatBuffer im = net.input.shallowClone();

        int m = n/groups;
        int k = size*size*c/groups;
        int n = outW*outH;

        int var1 = nweights/groups;
        int var2 = c/groups*h*w;

        for(i = 0; i < batch; ++i){
            for(j = 0; j < groups; ++j){

                _a.offset(j*var1);
                _c.offset((i*groups + j)*n*m);
                im.offset((i*groups + j)*var2);

                if (size == 1) {
                    Gemm.gemm(0,0,m,n,k,1,_a,k,im,n,1,_c,n);
                }
                else {
                    ImCol.im2ColCpu(im, c/groups, h, w, size, stride, pad, _b);
                    Gemm.gemm(0,0,m,n,k,1,_a,k,_b,n,1,_c,n);
                }
            }
        }

        if(batchNormalize != 0){
            BatchnormLayer.staticForward(this, net);
        }
        else {
            addBias(output, biases, batch, this.n, outH*outW);
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

                FloatBuffer _a = delta.offsetNew((i*groups + j)*m*k);
                FloatBuffer _b = net.workspace;
                FloatBuffer _c = weightUpdates.offsetNew(j*nweights/groups);

                FloatBuffer im  = net.input.offsetNew((i * groups + j)*c/groups*h*w);
                FloatBuffer imd = net.delta.offsetNew((i * groups + j)*c/groups*h*w);

                if(size == 1){
                    _b = im;
                }
                else {
                    ImCol.im2ColCpu(im, c/groups, h, w, size, stride, pad, _b);
                }

                Gemm.gemm(0,1,m,_n,k,1,_a,k,_b,k,1,_c,_n);

                if (net.delta != null) {

                    _a = weights.offsetNew(j*nweights/groups);
                    _b = delta.offsetNew((i*groups + j)*m*k);
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

//    public Image getConvolutionalWeight(int i) {
//
//        int h = size;
//        int w = size;
//        int _c = c/groups;
//
//        FloatBuffer fb = weights.offsetNew(i*h*w*_c);
//        return new Image(w,h,_c,fb);
//    }

//    public void rgbgrWeights() {
//
//        int i;
//        for(i = 0; i < n; ++i){
//            Image im = getConvolutionalWeight(i);
//            if (im.c == 3) {
//                im.rgbgr();
//            }
//        }
//    }
//
//    public void rescaleWeights(float scale, float trans) {
//
//        int i;
//        for(i = 0; i < n; ++i){
//            Image im = getConvolutionalWeight(i);
//            if (im.c == 3) {
//
//                im.scale(scale);
//                float sum = Util.sumArray(im.data, im.w*im.h*im.c);
//                biases.put(i,biases.get(i) + sum*trans);
//            }
//        }
//    }
//
//    public Image[] getWeights() {
//
//        Image[] imWeights = new Image[n];
//
//        int i;
//        for(i = 0; i < n; ++i){
//
//            imWeights[i] = getConvolutionalWeight(i).copyImage();
//            imWeights[i].normalize();
//        }
//        return imWeights;
//    }

//    public Image[] visualizeConvolutionalLayer(byte[] window, Image[] prevWeights)
//    {
//        return getWeights();
//    }
}
