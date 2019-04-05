package Layers;

import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Enums.Activation;
import Enums.LayerType;
import Tools.Blas;
import Tools.Buffers;
import org.lwjgl.BufferUtils;

public class RnnLayer extends Layer {

    public static void incrementLayer(Layer l, int steps) {

        int num = l.outputs*l.batch*steps;

        l.output = Buffers.offset(l.output,num);
        l.delta = Buffers.offset(l.delta,num);
        l.x = Buffers.offset(l.x,num);
        l.xNorm = Buffers.offset(l.xNorm,num);
    }

    public RnnLayer(int batch, int inputs, int outputs, int steps, Activation activation, int batch_normalize, int adam) {
        
        batch = batch / steps;
        this.batch = batch;
        this.type = LayerType.RNN;
        this.steps = steps;
        this.inputs = inputs;

        this.state = Buffers.newBufferF(batch*outputs);
        this.prevState = Buffers.newBufferF(batch*outputs);

        this.inputLayer = new ConnectedLayer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
        this.inputLayer.batch = batch;

        this.selfLayer = new ConnectedLayer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
        this.selfLayer.batch = batch;

        this.outputLayer = new ConnectedLayer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
        this.outputLayer.batch = batch;

        this.outputs = outputs;
        this.output = this.outputLayer.output;
        this.delta = this.outputLayer.delta;
    }

    public void update(UpdateArgs a) {

        ((ConnectedLayer)inputLayer).update(a);
        ((ConnectedLayer)selfLayer).update(a);
        ((ConnectedLayer)outputLayer).update(a);
    }

    public void forward(Network net) {
        
        Network s = net.tryClone();
        
        s.train = net.train;
        int i;

        Blas.fillCpu(this.outputs * this.batch * this.steps, 0, outputLayer.delta, 1);
        Blas.fillCpu(this.outputs * this.batch * this.steps, 0, selfLayer.delta, 1);
        Blas.fillCpu(this.outputs * this.batch * this.steps, 0, inputLayer.delta, 1);
        
        if(net.train != 0)  {
            Blas.fillCpu(this.outputs * this.batch, 0, this.state, 1);
        }

        for (i = 0; i < this.steps; ++i) {
            s.input = net.input;

            ((ConnectedLayer)inputLayer).forward(s);
            s.input = this.state;

            ((ConnectedLayer)selfLayer).forward(s);

            var old_state = this.state;

            if(net.train != 0) {

                this.state = Buffers.offset(this.state,outputs*batch);
            }
            if(this.shortcut != 0){
                Blas.copyCpu(this.outputs * this.batch, old_state, 1, this.state, 1);
            }else{
                Blas.fillCpu(this.outputs * this.batch, 0, this.state, 1);
            }
            Blas.axpyCpu(this.outputs * this.batch, 1, inputLayer.output, 1, this.state, 1);
            Blas.axpyCpu(this.outputs * this.batch, 1, selfLayer.output, 1, this.state, 1);

            s.input = this.state;

            ((ConnectedLayer)outputLayer).forward(s);

            net.input = Buffers.offset(net.input,this.inputs*this.batch);

            incrementLayer(inputLayer, 1);
            incrementLayer(selfLayer, 1);
            incrementLayer(outputLayer, 1);
        }
    }

    public void backward(Network net) {

        Network s = net.tryClone();
        s.train = net.train;
        int i;

        incrementLayer(inputLayer, this.steps-1);
        incrementLayer(selfLayer, this.steps-1);
        incrementLayer(outputLayer, this.steps-1);

        this.state = Buffers.offset(this.state,outputs*batch*steps);

        for (i = this.steps-1; i >= 0; --i) {
            Blas.copyCpu(this.outputs * this.batch, inputLayer.output, 1, this.state, 1);
            Blas.axpyCpu(this.outputs * this.batch, 1, selfLayer.output, 1, this.state, 1);

            s.input = this.state;
            s.delta = selfLayer.delta;

            ((ConnectedLayer)outputLayer).backward(s);

            this.state = Buffers.offset(this.state,- outputs*batch);

            s.input = this.state;

            s.delta = Buffers.offset(selfLayer.delta, - outputs*batch);

            if (i == 0) {
                s.delta = null;
            }

            ((ConnectedLayer)selfLayer).backward(s);

            Blas.copyCpu(this.outputs*this.batch, selfLayer.delta, 1, inputLayer.delta, 1);
            if (i > 0 && this.shortcut != 0) {

                var fb = Buffers.offset(selfLayer.delta, - outputs*batch);

                Blas.axpyCpu(this.outputs*this.batch, 1, selfLayer.delta, 1, fb, 1);
            }

            s.input = Buffers.offset(net.input,inputs*batch);

            if(net.delta != null) {

                s.delta = Buffers.offset(net.delta,inputs*batch);
            }
            else {
                s.delta = null;
            }

            ((ConnectedLayer)inputLayer).backward(s);

            incrementLayer(inputLayer, -1);
            incrementLayer(selfLayer, -1);
            incrementLayer(outputLayer, -1);
        }
    }
}
