package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Buffers;
import Yolo.Enums.LayerType;

public class UpsampleLayer extends Layer {

    public UpsampleLayer(int batch, int w, int h, int c, int stride) {

        this.type = LayerType.UPSAMPLE;
        this.batch = batch;
        this.w = w;
        this.h = h;
        this.c = c;
        this.outW = w*stride;
        this.outH = h*stride;
        this.outC = c;
        if(stride < 0){
            stride = -stride;
            this.reverse = 1;
            this.outW = w/stride;
            this.outH = h/stride;
        }
        this.stride = stride;
        this.outputs = this.outW*this.outH*this.outC;
        this.inputs = this.w*this.h*this.c;
        this.delta =  new FloatArray(this.outputs*batch);
        this.output = new FloatArray(this.outputs*batch);

        if(this.reverse != 0) {
            System.out.printf("Downs %3dX             %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, this.outW, this.outH, this.outC);
        }
        else {
            System.out.printf("Ups   %3dX             %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, this.outW, this.outH, this.outC);
        }

    }

    public void resize(int w, int h) {

        this.w = w;
        this.h = h;
        this.outW = w*this.stride;
        this.outH = h*this.stride;
        if(this.reverse != 0){
            this.outW = w/this.stride;
            this.outH = h/this.stride;
        }

        this.outputs = this.outW*this.outH*this.outC;
        this.inputs = this.h*this.w*this.c;
        this.delta =  Buffers.realloc(this.delta, this.outputs*this.batch);
        this.output = Buffers.realloc(this.output, this.outputs*this.batch);
    }

    public void forward(Network net) {

        this.output.setAll(0,outputs*batch);

        if(this.reverse != 0){
            Blas.upsampleCpu(this.output, this.outW, this.outH, this.c, this.batch, this.stride, 0, this.scale, net.input);
        }
        else{
            Blas.upsampleCpu(net.input, this.w, this.h, this.c, this.batch, this.stride, 1, this.scale, this.output);
        }
    }

    public void backward(Network net) {

        if(this.reverse != 0){
            Blas.upsampleCpu(this.delta, this.outW, this.outH, this.c, this.batch, this.stride, 1, this.scale, net.delta);
        }
        else{
            Blas.upsampleCpu(net.delta, this.w, this.h, this.c, this.batch, this.stride, 0, this.scale, this.delta);
        }
    }
}
