package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

public class ActivationLayer extends Layer {

    public ActivationLayer(int batch, int inputs, Activation activation) {

        this.type = LayerType.ACTIVE;

        this.inputs = inputs;
        this.outputs = inputs;
        this.batch = batch;

        this.output = new FloatArray(batch*inputs);
        this.delta = new FloatArray(batch*inputs);

        this.activation = activation;
    }

    public void forward(Network net) {

        net.input.copyInto(outputs*batch,output);
        Activation.activateArray(output, outputs*batch, activation);
    }

    public void backward(Network net) {

        Activation.gradientArray(output, outputs*batch, activation, delta);
        delta.copyInto(outputs*batch,net.delta);
    }

}
