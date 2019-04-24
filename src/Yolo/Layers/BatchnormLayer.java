package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Yolo.Enums.LayerType;

public class BatchnormLayer extends Layer {

    public BatchnormLayer(int batch, int w, int h, int c) {

        this.type = LayerType.BATCHNORM;
        this.batch = batch;
        this.h = this.outH = h;
        this.w = this.outW = w;
        this.c = this.outC = c;

        this.output = new FloatArray(h * w * c * batch);
        this.delta  = new FloatArray(h * w * c * batch);
        this.inputs = w*h*c;
        this.outputs = this.inputs;

        this.scales = new FloatArray(c);
        this.scaleUpdates = new FloatArray(c);
        this.biases = new FloatArray(c);
        this.biasUpdates = new FloatArray(c);
        
        for(int i = 0; i < c; ++i){

            this.scales.put(i,1);
        }

        this.mean = new FloatArray(c);
        this.variance = new FloatArray(c);

        this.rollingMean = new FloatArray(c);
        this.rollingVariance = new FloatArray(c);
    }

    public static void backwardScaleCpu(FloatArray xNorm, FloatArray delta, int batch, int n, int size, FloatArray scaleUpdates) {

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

    public static void meanDeltaCpu(FloatArray delta, FloatArray variance, int batch, int filters, int spatial, FloatArray meanDelta) {

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

    public static void varianceDeltaCpu(FloatArray x, FloatArray delta, FloatArray mean, FloatArray variance, int batch, int filters, int spatial, FloatArray varianceDelta) {

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

    public static void normalizeDeltaCpu(FloatArray x, FloatArray mean, FloatArray variance, FloatArray meanDelta,
                                         FloatArray varianceDelta, int batch, int filters, int spatial, FloatArray delta) {

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
        
        System.out.println("Resize batchnorm, Layer - not implemented");
    }

    public void forward(Network net) {

        if(type == LayerType.BATCHNORM) {
            net.input.copyInto(outputs*batch,output);
        }

        output.copyInto(outputs*batch,x);

        if(net.train != 0){
            Blas.meanCpu(output, batch, outC, outH*outW, mean);
            Blas.varianceCpu(output, mean, batch, outC, outH*outW, variance);

            Blas.scalCpu(outC, 0.99f, rollingMean, 1);
            Blas.axpyCpu(outC, 0.01f, mean, 1, rollingMean, 1);
            Blas.scalCpu(outC, 0.99f, rollingVariance, 1);
            Blas.axpyCpu(outC, 0.01f, variance, 1, rollingVariance, 1);

            Blas.normalizeCpu(output, mean, variance, batch, outC, outH*outW);
            output.copyInto(outputs*batch,xNorm);
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
            delta.copyInto(outputs*batch,net.delta);
        }
    }

    public static void staticForward(Layer l, Network net) {

        if(l.type == LayerType.BATCHNORM) {
            net.input.copyInto(l.outputs*l.batch,l.output);
        }

        l.output.copyInto(l.outputs*l.batch,l.x);
        
        if(net.train != 0){
            Blas.meanCpu(l.output, l.batch, l.outC, l.outH*l.outW, l.mean);
            Blas.varianceCpu(l.output, l.mean, l.batch, l.outC, l.outH*l.outW, l.variance);

            Blas.scalCpu(l.outC, 0.99f, l.rollingMean, 1);
            Blas.axpyCpu(l.outC, 0.01f, l.mean, 1, l.rollingMean, 1);
            Blas.scalCpu(l.outC, 0.99f, l.rollingVariance, 1);
            Blas.axpyCpu(l.outC, 0.01f, l.variance, 1, l.rollingVariance, 1);

            Blas.normalizeCpu(l.output, l.mean, l.variance, l.batch, l.outC, l.outH*l.outW);
            l.output.copyInto(l.outputs*l.batch,l.xNorm);
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
            l.delta.copyInto(l.outputs*l.batch,net.delta);
        }
    }
}
