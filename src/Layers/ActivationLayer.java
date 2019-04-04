package Layers;

import Classes.Layer;
import Classes.Network;
import Enums.Activation;
import Enums.LayerType;
import Tools.Blas;
import org.lwjgl.BufferUtils;

public class ActivationLayer extends Layer {

    public ActivationLayer(int batch, int inputs, Activation activation) {

        this.type = LayerType.ACTIVE;

        this.inputs = inputs;
        this.outputs = inputs;
        this.batch = batch;

        this.output = BufferUtils.createFloatBuffer(batch*inputs);
        this.delta = BufferUtils.createFloatBuffer(batch*inputs);

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
