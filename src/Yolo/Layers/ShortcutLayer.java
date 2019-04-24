package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Buffers;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

public class ShortcutLayer extends Layer {

    public ShortcutLayer(int batch, int index, int w, int h, int c, int w2, int h2, int c2) {

        this.type = LayerType.SHORTCUT;
        this.batch = batch;
        this.w = w2;
        this.h = h2;
        this.c = c2;
        this.outW = w;
        this.outH = h;
        this.outC = c;
        this.outputs = w*h*c;
        this.inputs = this.outputs;

        this.index = index;

        this.delta = new FloatArray(this.outputs*batch);
        this.output = new FloatArray(this.outputs*batch);

        System.out.printf("res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n",index, w2,h2,c2, w,h,c);
    }

    public void resize(int w, int h) {

        assert(this.w == this.outW);
        assert(this.h == this.outH);

        this.w = this.outW = w;
        this.h = this.outH = h;
        this.outputs = w*h*this.outC;
        this.inputs = this.outputs;

        this.delta =  Buffers.realloc(this.delta, this.outputs*this.batch);
        this.output = Buffers.realloc(this.output, this.outputs*this.batch);
    }

    public void forward(Network net) {

        net.input.copyInto(outputs*batch,output);
        Blas.shortcutCpu(this.batch, this.w, this.h, this.c, net.layers[this.index].output, this.outW, this.outH, this.outC, this.alpha, this.beta, this.output);
        Activation.activateArray(this.output, this.outputs*this.batch, this.activation);
    }

    public void backward(Network net) {

        Activation.gradientArray(this.output, this.outputs*this.batch, this.activation, this.delta);
        Blas.axpyCpu(this.outputs*this.batch, this.alpha, this.delta, 1, net.delta, 1);
        Blas.shortcutCpu(this.batch, this.outW, this.outH, this.outC, this.delta, this.w, this.h, this.c, 1, this.beta, net.layers[this.index].delta);
    }
}
