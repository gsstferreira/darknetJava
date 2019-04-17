package Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Enums.Activation;
import Enums.LayerType;
import Tools.Blas;

public class RnnLayer extends Layer {

    public static void incrementLayer(Layer l, int steps) {

        int num = l.outputs*l.batch*steps;

        l.output.offset(num);
        l.delta.offset(num);
        l.x.offset(num);
        l.xNorm.offset(num);
    }

    public RnnLayer(int batch, int inputs, int outputs, int steps, Activation activation, int batch_normalize, int adam) {

        System.out.printf("RNN Layer: %d inputs, %d outputs\n", inputs, outputs);
        batch = batch / steps;
        this.batch = batch;
        this.type = LayerType.RNN;
        this.steps = steps;
        this.inputs = inputs;

        this.state = new FloatBuffer(batch*outputs);
        this.prevState = new FloatBuffer(batch*outputs);

        this.inputLayer = new ConnectedLayer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
        System.out.printf("\t\t");
        this.inputLayer.batch = batch;

        this.selfLayer = new ConnectedLayer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
        System.out.printf("\t\t");
        this.selfLayer.batch = batch;

        this.outputLayer = new ConnectedLayer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
        System.out.printf("\t\t");
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

            FloatBuffer old_state = this.state;

            if(net.train != 0) {

                this.state.offset(outputs*batch);
            }
            if(this.shortcut != 0){
                Blas.copyCpu(this.outputs * this.batch, old_state, 1, this.state, 1);
            }
            else{
                Blas.fillCpu(this.outputs * this.batch, 0, this.state, 1);
            }
            Blas.axpyCpu(this.outputs * this.batch, 1, inputLayer.output, 1, this.state, 1);
            Blas.axpyCpu(this.outputs * this.batch, 1, selfLayer.output, 1, this.state, 1);

            s.input = this.state;

            ((ConnectedLayer)outputLayer).forward(s);

            net.input.offset(this.inputs*this.batch);

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

        this.state.offset(outputs*batch*steps);

        for (i = this.steps-1; i >= 0; --i) {
            Blas.copyCpu(this.outputs * this.batch, inputLayer.output, 1, this.state, 1);
            Blas.axpyCpu(this.outputs * this.batch, 1, selfLayer.output, 1, this.state, 1);

            s.input = this.state;
            s.delta = selfLayer.delta;

            ((ConnectedLayer)outputLayer).backward(s);

            this.state.offset(-outputs*batch);

            s.input = this.state;

            s.delta = selfLayer.delta.offsetNew(-outputs*batch);

            if (i == 0) {
                s.delta = null;
            }

            ((ConnectedLayer)selfLayer).backward(s);

            Blas.copyCpu(this.outputs*this.batch, selfLayer.delta, 1, inputLayer.delta, 1);
            if (i > 0 && this.shortcut != 0) {

                FloatBuffer fb = selfLayer.delta.offsetNew(-outputs*batch);
                Blas.axpyCpu(this.outputs*this.batch, 1, selfLayer.delta, 1, fb, 1);
            }

            s.input = net.input.offsetNew(inputs*batch);

            if(net.delta != null) {

                s.delta = net.delta.offsetNew(inputs*batch);
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
