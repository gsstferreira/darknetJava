package Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Enums.Activation;
import Enums.LayerType;
import Tools.Blas;

public class ActivationLayer extends Layer {

    public ActivationLayer(int batch, int inputs, Activation activation) {

        this.type = LayerType.ACTIVE;

        this.inputs = inputs;
        this.outputs = inputs;
        this.batch = batch;

        this.output = new FloatBuffer(batch*inputs);
        this.delta = new FloatBuffer(batch*inputs);

        this.activation = activation;
    }

    public void forward(Network net) {

        Blas.copyCpu(outputs * batch, net.input, 1, output, 1);
        Activation.activateArray(output, outputs*batch, activation);
    }

    public void backward(Network net) {

        Activation.gradientArray(output, outputs*batch, activation, delta);
        Blas.copyCpu(outputs*batch, delta, 1, net.delta, 1);
    }

}
