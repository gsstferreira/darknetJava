package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Util;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

public class LogisticLayer extends Layer {

    public LogisticLayer(int batch, int inputs) {

        this.type = LayerType.LOGXENT;
        this.batch = batch;
        this.inputs = inputs;
        this.outputs = inputs;
        this.loss = new FloatArray(inputs*batch);
        this.output = new FloatArray(inputs*batch);
        this.delta = new FloatArray(inputs*batch);
        this.cost = new FloatArray(1);
    }

    public void forward(Network net) {

        net.input.copyInto(outputs*batch,this.output);
        Activation.activateArray(this.output, this.outputs*this.batch, Activation.LOGISTIC);
        if(net.truth != null){
            Blas.logisticXEntCpu(this.batch*this.inputs, this.output, net.truth, this.delta, this.loss);
            this.cost.put(0, Util.sumArray(this.loss, this.batch*this.inputs));
        }
    }

    public void backward(Network net) {
        
        Blas.axpyCpu(this.inputs*this.batch, 1, this.delta, 1, net.delta, 1);
    }
}
