package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Yolo.Enums.LayerType;
import Tools.Blas;

public class BatchnormLayer extends Layer {

    public BatchnormLayer(int batch, int w, int h, int c) {

        this.type = LayerType.BATCHNORM;
        this.batch = batch;
        this.h = this.outH = h;
        this.w = this.outW = w;
        this.c = this.outC = c;

        this.output = new FloatBuffer(h * w * c * batch);
        this.delta  = new FloatBuffer(h * w * c * batch);
        this.inputs = w*h*c;
        this.outputs = this.inputs;

        this.scales = new FloatBuffer(c);
        this.scaleUpdates = new FloatBuffer(c);
        this.biases = new FloatBuffer(c);
        this.biasUpdates = new FloatBuffer(c);
        
        for(int i = 0; i < c; ++i){

            this.scales.put(i,1);
        }

        this.mean = new FloatBuffer(c);
        this.variance = new FloatBuffer(c);

        this.rollingMean = new FloatBuffer(c);
        this.rollingVariance = new FloatBuffer(c);
    }

    public static void backwardScaleCpu(FloatBuffer xNorm, FloatBuffer delta, int batch, int n, int size, FloatBuffer scaleUpdates) {

        int i,b,f;
        for(f = 0; f < n; ++f){
            float sum = 0;
            for(b = 0; b < batch; ++b){
                for(i = 0; i < size; ++i){
                    int index = i + size*(f + n*b);

                    sum += delta.get(index) * xNorm.get(index);
                }
            }
            scaleUpdates.put(f,scaleUpdates.get(f) + sum);
        }
    }

    public static void meanDeltaCpu(FloatBuffer delta, FloatBuffer variance, int batch, int filters, int spatial, FloatBuffer meanDelta) {

        int i,j,k;
        for(i = 0; i < filters; ++i){

            meanDelta.put(i,0);
            for (j = 0; j < batch; ++j) {
                for (k = 0; k < spatial; ++k) {

                    int index = j*filters*spatial + i*spatial + k;
                    meanDelta.put(i,meanDelta.get(i) + delta.get(index));
                }
            }

            float val =  (float) (meanDelta.get(i) * (-1/Math.sqrt(variance.get(i) + 0.00001f)));
            meanDelta.put(i,val);
        }
    }

    public static void varianceDeltaCpu(FloatBuffer x, FloatBuffer delta, FloatBuffer mean, FloatBuffer variance, int batch, int filters, int spatial, FloatBuffer varianceDelta) {

        int i,j,k;
        for(i = 0; i < filters; ++i){

            varianceDelta.put(i,0);
            for(j = 0; j < batch; ++j){
                for(k = 0; k < spatial; ++k){
                    int index = j*filters*spatial + i*spatial + k;

                    float val = varianceDelta.get(i) + delta.get(index) * (x.get(index) - mean.get(i));
                    varianceDelta.put(i,val);
                }
            }

            float val = varianceDelta.get(i) * (float) (Math.pow(variance.get(i) + 0.00001f, -1.5f))/-2;
            varianceDelta.put(i,val);
        }
    }

    public static void normalizeDeltaCpu(FloatBuffer x, FloatBuffer mean, FloatBuffer variance, FloatBuffer meanDelta,
                                         FloatBuffer varianceDelta, int batch, int filters, int spatial, FloatBuffer delta) {

        int f, j, k;
        for(j = 0; j < batch; ++j){
            for(f = 0; f < filters; ++f){
                for(k = 0; k < spatial; ++k){
                    int index = j*filters*spatial + f*spatial + k;

                    double val = delta.get(index) * 1/(Math.sqrt(variance.get(f) + 0.00001f));
                    val += varianceDelta.get(f) * 2 * (x.get(index) - mean.get(f))/(spatial * batch) + meanDelta.get(f)/(spatial * batch);

                    delta.put(index,(float)val);
                }
            }
        }
    }

    public void resize(int w, int h) {
        
        System.out.println("Resize batchnor, Layer - not implemented");
    }

    public void forward(Network net) {

        if(type == LayerType.BATCHNORM) {
            Blas.copyCpu(outputs*batch, net.input, 1, output, 1);
        }

        Blas.copyCpu(outputs*batch, output, 1, x, 1);

        if(net.train != 0){
            Blas.meanCpu(output, batch, outC, outH*outW, mean);
            Blas.varianceCpu(output, mean, batch, outC, outH*outW, variance);

            Blas.scalCpu(outC, 0.99f, rollingMean, 1);
            Blas.axpyCpu(outC, 0.01f, mean, 1, rollingMean, 1);
            Blas.scalCpu(outC, 0.99f, rollingVariance, 1);
            Blas.axpyCpu(outC, 0.01f, variance, 1, rollingVariance, 1);

            Blas.normalizeCpu(output, mean, variance, batch, outC, outH*outW);
            Blas.copyCpu(outputs*batch, output, 1, xNorm, 1);
        }
        else {
            Blas.normalizeCpu(output, rollingMean, rollingVariance, batch, outC, outH*outW);
        }
        ConvolutionalLayer.scaleBias(output, scales, batch, outC, outH*outW);
        ConvolutionalLayer.addBias(output, biases, batch, outC, outH*outW);
    }

    public void backward(Network net) {

        if(net.train == 0){
            mean = rollingMean;
            variance = rollingVariance;
        }

        ConvolutionalLayer.backwardBias(biasUpdates, delta, batch, outC, outW*outH);
        backwardScaleCpu(xNorm, delta, batch, outC, outW*outH, scaleUpdates);

        ConvolutionalLayer.scaleBias(delta, scales, batch, outC, outH*outW);

        meanDeltaCpu(delta, variance, batch, outC, outW*outH, meanDelta);
        varianceDeltaCpu(x, delta, mean, variance, batch, outC, outW*outH, varianceDelta);
        normalizeDeltaCpu(x, mean, variance, meanDelta, varianceDelta, batch, outC, outW*outH, delta);

        if(type == LayerType.BATCHNORM) {
            Blas.copyCpu(outputs*batch, delta, 1, net.delta, 1);
        }
    }

    public static void staticForward(Layer l, Network net) {

        if(l.type == LayerType.BATCHNORM) {
            Blas.copyCpu(l.outputs*l.batch, net.input, 1, l.output, 1);
        }
        
        Blas.copyCpu(l.outputs*l.batch, l.output, 1, l.x, 1);
        
        if(net.train != 0){
            Blas.meanCpu(l.output, l.batch, l.outC, l.outH*l.outW, l.mean);
            Blas.varianceCpu(l.output, l.mean, l.batch, l.outC, l.outH*l.outW, l.variance);

            Blas.scalCpu(l.outC, 0.99f, l.rollingMean, 1);
            Blas.axpyCpu(l.outC, 0.01f, l.mean, 1, l.rollingMean, 1);
            Blas.scalCpu(l.outC, 0.99f, l.rollingVariance, 1);
            Blas.axpyCpu(l.outC, 0.01f, l.variance, 1, l.rollingVariance, 1);

            Blas.normalizeCpu(l.output, l.mean, l.variance, l.batch, l.outC, l.outH*l.outW);
            Blas.copyCpu(l.outputs*l.batch, l.output, 1, l.xNorm, 1);
        } 
        else {
            Blas.normalizeCpu(l.output, l.rollingMean, l.rollingVariance, l.batch, l.outC, l.outH*l.outW);
        }
        ConvolutionalLayer.scaleBias(l.output, l.scales, l.batch, l.outC, l.outH*l.outW);
        ConvolutionalLayer.addBias(l.output, l.biases, l.batch, l.outC, l.outH*l.outW);
    }

    public static void staticBackward(Layer l, Network net) {

        if(net.train == 0){
            l.mean = l.rollingMean;
            l.variance = l.rollingVariance;
        }

        ConvolutionalLayer.backwardBias(l.biasUpdates, l.delta, l.batch, l.outC, l.outW*l.outH);
        backwardScaleCpu(l.xNorm, l.delta, l.batch, l.outC, l.outW*l.outH, l.scaleUpdates);

        ConvolutionalLayer.scaleBias(l.delta, l.scales, l.batch, l.outC, l.outH*l.outW);

        meanDeltaCpu(l.delta, l.variance, l.batch, l.outC, l.outW*l.outH, l.meanDelta);
        varianceDeltaCpu(l.x, l.delta, l.mean, l.variance, l.batch, l.outC, l.outW*l.outH, l.varianceDelta);
        normalizeDeltaCpu(l.x, l.mean, l.variance, l.meanDelta, l.varianceDelta, l.batch, l.outC, l.outW*l.outH, l.delta);
        
        if(l.type == LayerType.BATCHNORM) {
            Blas.copyCpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
        }
    }
}
