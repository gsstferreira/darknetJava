package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Yolo.Enums.LayerType;

public class L2NormLayer extends Layer {

    public L2NormLayer(int batch, int inputs) {

        this.type = LayerType.L2NORM;
        this.batch = batch;
        this.inputs = inputs;
        this.outputs = inputs;
        this.output = new FloatArray(inputs*batch);
        this.scales = new FloatArray(inputs*batch);
        this.delta = new FloatArray(inputs*batch);
    }

    public void forward(Network net) {

        net.input.copyInto(outputs*batch,this.output);
        Blas.l2normalizeCpu(this.output, this.scales, this.batch, this.outC, this.outW*this.outH);
    }

    public void backward(Network net) {

        Blas.axpyCpu(this.inputs*this.batch, 1, this.delta, 1, net.delta, 1);
    }
}
