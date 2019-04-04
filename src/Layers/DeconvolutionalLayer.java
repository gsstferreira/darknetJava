package Layers;

import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Enums.Activation;
import Enums.LayerType;
import Tools.*;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

public class DeconvolutionalLayer extends Layer {

    private int getWorkspaceSize(){
        
        return h*w*size*size*n;
    }

    public void bilinearInit() {
        
        int i,j,f;
        float center = (size-1) / 2.0f;
        for(f = 0; f < n; ++f){
            for(j = 0; j < size; ++j){
                for(i = 0; i < size; ++i){
                    float val = (1 - Math.abs(i - center)) * (1 - Math.abs(j - center));
                    int _c = f%c;
                    int ind = f*size*size*c + _c*size*size + j*size + i;
                    
                    weights.put(ind,val);
                }
            }
        }
    }
    
    public DeconvolutionalLayer(int batch, int h, int w, int c, int n, int size, int stride, int padding, Activation activation, int batch_normalize, int adam) {
        
        int i;

        this.type = LayerType.DECONVOLUTIONAL;

        this.h = h;
        this.w = w;
        this.c = c;
        this.n = n;
        this.batch = batch;
        this.stride = stride;
        this.size = size;

        this.nweights = c*n*size*size;
        this.nbiases = n;

        this.weights = BufferUtils.createFloatBuffer(c*n*size*size);
        this.weightUpdates = BufferUtils.createFloatBuffer(c*n*size*size);

        this.biases = BufferUtils.createFloatBuffer(n);
        this.biasUpdates = BufferUtils.createFloatBuffer(n);

        float scale = 0.02f;
        for(i = 0; i < c*n*size*size; ++i) {

            this.weights.put(i,scale* Rand.randNormal());
        }

        for(i = 0; i < n; ++i){

            this.biases.put(i,0);
        }
        
        this.pad = padding;

        this.outH = (this.h - 1) * this.stride + this.size - 2*this.pad;
        this.outW = (this.w - 1) * this.stride + this.size - 2*this.pad;
        this.outW = n;
        this.outputs = this.outW * this.outH * this.outC;
        this.inputs = this.w * this.h * this.c;

        Blas.scalCpu(this.nweights, (float) this.outW * this.outH / (this.w * this.h), this.weights, 1);

        this.output = BufferUtils.createFloatBuffer(this.batch*this.outputs);
        this.delta  = BufferUtils.createFloatBuffer(this.batch*this.outputs);
        
        this.batchNormalize = batch_normalize;

        if(batch_normalize != 0){
            
            this.scales = BufferUtils.createFloatBuffer(n);
            this.scaleUpdates = BufferUtils.createFloatBuffer(n);
            
            for(i = 0; i < n; ++i){
                
                this.scales.put(i,1);
            }

            this.mean = BufferUtils.createFloatBuffer(n);
            this.variance = BufferUtils.createFloatBuffer(n);

            this.meanDelta = BufferUtils.createFloatBuffer(n);
            this.varianceDelta = BufferUtils.createFloatBuffer(n);

            this.rollingMean = BufferUtils.createFloatBuffer(n);
            this.rollingVariance = BufferUtils.createFloatBuffer(n);
            
            this.x = BufferUtils.createFloatBuffer(this.batch*this.outputs);
            this.xNorm = BufferUtils.createFloatBuffer(this.batch*this.outputs);
        }
        
        if(adam != 0){
            this.m = BufferUtils.createFloatBuffer(c*n*size*size);
            this.v = BufferUtils.createFloatBuffer(c*n*size*size);
            this.biasM = BufferUtils.createFloatBuffer(n);
            this.scaleM = BufferUtils.createFloatBuffer(n);
            this.biasV = BufferUtils.createFloatBuffer(n);
            this.scaleV = BufferUtils.createFloatBuffer(n);
        }
        
        this.activation = activation;
        this.workspaceSize = getWorkspaceSize();
    }

    public void denormalize() {
        
        int i, j;
        for(i = 0; i < this.n; ++i){
            
            float scale = scales.get(i)/(float)Math.sqrt(rollingVariance.get(i) + 0.00001);

            for(j = 0; j < this.c*this.size*this.size; ++j){
                
                float val = weights.get(i*c*size*size + j) * scale;
                weights.put(i*c*size*size + j,val);
            }
            
            biases.put(i, biases.get(i) - rollingMean.get(i)*scale);
            scales.put(i,1);
            rollingMean.put(i,0);
            rollingVariance.put(i,1);
        }
    }

    public void resize(int height, int width) {
        
        h = height;
        w = width;
        outH = (h - 1) * stride + size - 2*pad;
        outW = (w - 1) * stride + size - 2*pad;

        outputs = outH * outW * outC;
        inputs = w * h * c;

        output = BufferUtil.reallocBuffer(output,batch*outputs);
        delta  = BufferUtil.reallocBuffer(delta,batch*outputs);
        
        if(batchNormalize != 0){
            x = BufferUtil.reallocBuffer(x,batch*outputs);
            xNorm = BufferUtil.reallocBuffer(xNorm,batch*outputs);
        }
        
        workspaceSize = getWorkspaceSize();
    }

    public void forward(Network net) {

        int i;

        int m = this.size*this.size*this.n;
        int n = this.h*this.w;
        int k = this.c;

        Blas.fillCpu(this.outputs*this.batch, 0, this.output, 1);

        for(i = 0; i < this.batch; ++i){
            FloatBuffer a = this.weights;
            FloatBuffer b = BufferUtil.offsetBuffer(net.input,i*this.c*this.h*this.w);
            FloatBuffer c = net.workspace;

            Gemm.gemmCpu(1,0,m,n,k,1,a,m,b,n,0,c,n);

            FloatBuffer fb = BufferUtil.offsetBuffer(this.output,i*this.outputs);
            ImCol.col2ImCpu(net.workspace, this.outC, this.outH, this.outW, this.size, this.stride, this.pad, fb);
        }

        if (this.batchNormalize != 0) {

            BatchnormLayer.staticForward(this,net);
        }
        else {
            ConvolutionalLayer.addBias(this.output, this.biases, this.batch, this.n, this.outW*this.outH);
        }
        Activation.activateArray(this.output, this.batch*this.n*this.outW*this.outH, this.activation);
    }

    public void backward(Network net) {

        int i;

        Activation.gradientArray(this.output, this.outputs*this.batch, this.activation, this.delta);

        if(this.batchNormalize != 0){

            BatchnormLayer.staticBackward(this,net);
        }

        else {
            ConvolutionalLayer.backwardBias(this.biasUpdates, this.delta, this.batch, this.n, this.outW*this.outH);
        }

        for(i = 0; i < this.batch; ++i){

            int m = this.c;
            int n = this.size*this.size*this.n;
            int k = this.h*this.w;

            FloatBuffer a = BufferUtil.offsetBuffer(net.input,i*m*k);
            FloatBuffer b = net.workspace;
            FloatBuffer c = this.weightUpdates;

            FloatBuffer fb = BufferUtil.offsetBuffer(delta,i*this.outputs);

            ImCol.im2ColCpu(fb, this.outC, this.outH, this.outW,this.size, this.stride, this.pad, b);
            Gemm.gemmCpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if(net.delta != null){

                m = this.c;
                n = this.h*this.w;
                k = this.size*this.size*this.n;

                a = this.weights;
                b = net.workspace;
                c = BufferUtil.offsetBuffer(net.delta,i*n*m);

                Gemm.gemmCpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
            }
        }
    }

    public void update(UpdateArgs a) {

        float learning_rate = a.learningRate*this.learningRateScale;
        float momentum = a.momentum;
        float decay = a.decay;
        int batch = a.batch;

        int size = this.size*this.size*this.c*this.n;

        Blas.axpyCpu(this.n, learning_rate/batch, this.biasUpdates, 1, this.biases, 1);
        Blas.scalCpu(this.n, momentum, this.biasUpdates, 1);

        if(this.scales != null){

            Blas.axpyCpu(this.n, learning_rate/batch, this.scaleUpdates, 1, this.scales, 1);
            Blas.scalCpu(this.n, momentum, this.scaleUpdates, 1);
        }

        Blas.axpyCpu(size, -decay*batch, this.weights, 1, this.weightUpdates, 1);
        Blas.axpyCpu(size, learning_rate/batch, this.weightUpdates, 1, this.weights, 1);
        Blas.scalCpu(size, momentum, this.weightUpdates, 1);
    }
}