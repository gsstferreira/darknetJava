package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Tools.Blas;
import Tools.Gemm;
import Tools.Rand;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;


public class ConnectedLayer extends Layer {

    public ConnectedLayer(int batch, int inputs, int outputs, Activation activation, int batch_normalize, int adam) {
        
        int i;
        this.learningRateScale = 1;
        this.type = LayerType.CONNECTED;

        this.inputs = inputs;
        this.outputs = outputs;
        this.batch=batch;
        this.batchNormalize = batch_normalize;
        this.h = 1;
        this.w = 1;
        this.c = inputs;
        this.outH = 1;
        this.outW = 1;
        this.outC = outputs;

        this.output = new FloatBuffer(batch*outputs);
        this.delta = new FloatBuffer(batch*outputs);

        this.weightUpdates = new FloatBuffer(inputs*outputs);
        this.biasUpdates = new FloatBuffer(outputs);

        this.weights = new FloatBuffer(inputs*outputs);
        this.biases = new FloatBuffer(outputs);

        float scale = (float) Math.sqrt(2./inputs);

        for(i = 0; i < outputs*inputs; ++i){

            this.weights.put(i,scale * Rand.randUniform(-1,1));
        }

        for(i = 0; i < outputs; ++i){

            this.biases.put(i,0);
        }

        if(adam != 0){
            
            this.m = new FloatBuffer(this.inputs*this.outputs);
            this.v = new FloatBuffer(this.inputs*this.outputs);
            this.biasM = new FloatBuffer(this.outputs);
            this.scaleM = new FloatBuffer(this.outputs);
            this.biasV = new FloatBuffer(this.outputs);
            this.scaleV = new FloatBuffer(this.outputs);
        }

        if(batch_normalize != 0){
            this.scales = new FloatBuffer(outputs);
            this.scaleUpdates = new FloatBuffer(outputs);
            
            for(i = 0; i < outputs; ++i){
                this.scales.put(i,1);
            }

            this.mean = new FloatBuffer(outputs);
            this.meanDelta = new FloatBuffer(outputs);
            this.variance = new FloatBuffer(outputs);
            this.varianceDelta = new FloatBuffer(outputs);

            this.rollingMean = new FloatBuffer(outputs);
            this.rollingVariance = new FloatBuffer(outputs);

            this.x = new FloatBuffer(batch*outputs);
            this.xNorm = new FloatBuffer(batch*outputs);
        }
        
        this.activation = activation;
    }

    public void update(UpdateArgs a) {
        
        float learning_rate = a.learningRate*this.learningRateScale;
        float momentum = a.momentum;
        float decay = a.decay;
        int batch = a.batch;

        Blas.axpyCpu(this.outputs, learning_rate / batch, this.biasUpdates, 1, this.biases, 1);
        Blas.scalCpu(this.outputs, momentum, this.biasUpdates, 1);

        if(this.batchNormalize != 0){
            Blas.axpyCpu(this.outputs, learning_rate/batch, this.scaleUpdates, 1, this.scales, 1);
            Blas.scalCpu(this.outputs, momentum, this.scaleUpdates, 1);
        }

        Blas.axpyCpu(this.inputs*this.outputs, -decay*batch, this.weights, 1, this.weightUpdates, 1);
        Blas.axpyCpu(this.inputs*this.outputs, learning_rate/batch, this.weightUpdates, 1, this.weights, 1);
        Blas.scalCpu(this.inputs*this.outputs, momentum, this.weightUpdates, 1);
    }

    public void forward(Network net) {
        
        Blas.fillCpu(this.outputs*this.batch, 0, this.output, 1);
        
        int m = this.batch;
        int k = this.inputs;
        int n = this.outputs;
        FloatBuffer a = net.input;
        FloatBuffer b = this.weights;
        FloatBuffer c = this.output;

        Gemm.gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(this.batchNormalize != 0){
            BatchnormLayer.staticForward(this, net);
        }
        else {
            ConvolutionalLayer.addBias(this.output, this.biases, this.batch, this.outputs, 1);
        }
        Activation.activateArray(this.output, this.outputs*this.batch, this.activation);
    }

    public void backward(Network net) {

        Activation.gradientArray(this.output, this.outputs*this.batch, this.activation, this.delta);

        if(this.batchNormalize != 0){
            BatchnormLayer.staticBackward(this, net);
        } else {
            ConvolutionalLayer.backwardBias(this.biasUpdates, this.delta, this.batch, this.outputs, 1);
        }

        int m = this.outputs;
        int k = this.batch;
        int n = this.inputs;

        FloatBuffer a = this.delta;
        FloatBuffer b = net.input;
        FloatBuffer c = this.weightUpdates;

        Gemm.gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

        m = this.batch;
        k = this.outputs;
        n = this.inputs;

        a = this.delta;
        b = this.weights;
        c = net.delta;

        if(c != null) {
            Gemm.gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

//    public void denormalize() {
//
//        int i, j;
//        for(i = 0; i < this.outputs; ++i){
//
//            float scale = scales.get(i)/(float)Math.sqrt(rollingVariance.get(i) + 0.000001);
//
//            for(j = 0; j < this.inputs; ++j){
//
//                weights.put(i*inputs + j,weights.get(i*inputs + j) * scale);
//            }
//
//            float val = biases.get(i) - rollingMean.get(i) * scale;
//            biases.put(i,val);
//
//            scales.put(i,1);
//
//            rollingMean.put(i,0);
//            rollingVariance.put(i,1);
//        }
//    }
    
}
