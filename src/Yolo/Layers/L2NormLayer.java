package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
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
        this.output = new FloatBuffer(inputs*batch);
        this.scales = new FloatBuffer(inputs*batch);
        this.delta = new FloatBuffer(inputs*batch);
    }

    public void forward(Network net) {
        
        Blas.copyCpu(this.outputs*this.batch, net.input, 1, this.output, 1);
        Blas.l2normalizeCpu(this.output, this.scales, this.batch, this.outC, this.outW*this.outH);
    }

    public void backward(Network net) {

        Blas.axpyCpu(this.inputs*this.batch, 1, this.delta, 1, net.delta, 1);
    }
}
