package Layers;

import Classes.Layer;
import Classes.Network;
import Enums.Activation;
import Enums.LayerType;
import Tools.Blas;
import Tools.Buffers;
import Tools.Util;

public class LogisticLayer extends Layer {

    public LogisticLayer(int batch, int inputs) {

        this.type = LayerType.LOGXENT;
        this.batch = batch;
        this.inputs = inputs;
        this.outputs = inputs;
        this.loss = Buffers.newBufferF(inputs*batch);
        this.output = Buffers.newBufferF(inputs*batch);
        this.delta = Buffers.newBufferF(inputs*batch);
        this.cost = Buffers.newBufferF(1);
    }

    public void forward(Network net) {

        Blas.copyCpu(this.outputs * this.batch, net.input, 1, this.output, 1);
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
