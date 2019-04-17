package Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Enums.LayerType;
import Tools.Blas;
import Tools.Buffers;

public class ReorgLayer extends Layer {

    public ReorgLayer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra) {

        this.type = LayerType.REORG;
        this.batch = batch;
        this.stride = stride;
        this.extra = extra;
        this.h = h;
        this.w = w;
        this.c = c;
        this.flatten = flatten;
        if(reverse != 0){
            this.outW = w*stride;
            this.outH = h*stride;
            this.outC = c/(stride*stride);
        }else{
            this.outW = w/stride;
            this.outH = h/stride;
            this.outC = c*(stride*stride);
        }
        this.reverse = reverse;

        this.outputs = this.outH * this.outW * this.outC;
        this.inputs = h*w*c;

        if(this.extra != 0){
            this.outW = this.outH = this.outC = 0;
            this.outputs = this.inputs + this.extra;
        }

        if(extra != 0){
            System.out.printf("reorg              %4d   ->  %4d\n",  this.inputs, this.outputs);
        }
        else {
            System.out.printf("reorg              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  stride, w, h, c, this.outW, this.outH, this.outC);
        }

        int output_size = this.outputs * batch;
        this.output = new FloatBuffer(output_size);
        this.delta = new FloatBuffer(output_size);
    }

    public void resize(int w, int h) {

        int stride = this.stride;
        int c = this.c;

        this.h = h;
        this.w = w;

        if(this.reverse != 0){
            this.outW = w*stride;
            this.outH = h*stride;
            this.outC = c/(stride*stride);
        }

        else{
            this.outW = w/stride;
            this.outH = h/stride;
            this.outC = c*(stride*stride);
        }

        this.outputs = this.outH * this.outW * this.outC;
        this.inputs = this.outputs;
        int output_size = this.outputs * this.batch;

        this.output = Buffers.realloc(this.output, output_size);
        this.delta = Buffers.realloc(this.delta, output_size);
    }

    public void forward(Network net) {

        int i;
        if(this.flatten != 0){

            Buffers.copy(net.input,this.output,this.outputs*this.batch);

            if(this.reverse != 0){
                Blas.flatten(this.output, this.w*this.h, this.c, this.batch, 0);
            }
            else{
                Blas.flatten(this.output, this.w*this.h, this.c, this.batch, 1);
            }
        }
        else if (this.extra != 0) {
            for(i = 0; i < this.batch; ++i){

                var fb = net.input.offsetNew(i*this.inputs);
                var fb2 = this.output.offsetNew(i*this.outputs);

                Blas.copyCpu(this.inputs, fb, 1, fb2, 1);
            }
        }
        else if (this.reverse != 0){
            Blas.reorgCpu(net.input, this.w, this.h, this.c, this.batch, this.stride, 1, this.output);
        }
        else {
            Blas.reorgCpu(net.input, this.w, this.h, this.c, this.batch, this.stride, 0, this.output);
        }
    }

    public void backward(Network net) {

        int i;
        if(this.flatten != 0){

            Buffers.copy(this.delta,net.delta,this.outputs*this.batch);

            if(this.reverse != 0){
                Blas.flatten(net.delta, this.w*this.h, this.c, this.batch, 1);
            }
            else{
                Blas.flatten(net.delta, this.w*this.h, this.c, this.batch, 0);
            }
        }
        else if(this.reverse!= 0) {
            Blas.reorgCpu(this.delta, this.w, this.h, this.c, this.batch, this.stride, 0, net.delta);
        }
        else if (this.extra != 0) {
            for(i = 0; i < this.batch; ++i){

                var fb = this.delta.offsetNew(i*this.outputs);
                var fb2 = net.delta.offsetNew(i*this.inputs);

                Blas.copyCpu(this.inputs, fb, 1, fb2, 1);
            }
        }
        else{
            Blas.reorgCpu(this.delta, this.w, this.h, this.c, this.batch, this.stride, 1, net.delta);
        }
    }
}
