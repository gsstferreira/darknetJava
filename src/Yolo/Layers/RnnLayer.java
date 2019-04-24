package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Tools.Blas;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

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

        this.state = new FloatArray(batch*outputs);
        this.prevState = new FloatArray(batch*outputs);

        this.inputLayer = new ConnectedLayer(batch*steps, inputs, outputs, activation, batch_normalize, adam);
        System.out.print("\t\t");
        this.inputLayer.batch = batch;

        this.selfLayer = new ConnectedLayer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
        System.out.print("\t\t");
        this.selfLayer.batch = batch;

        this.outputLayer = new ConnectedLayer(batch*steps, outputs, outputs, activation, batch_normalize, adam);
        System.out.print("\t\t");
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

            FloatArray oldState = this.state;

            if(net.train != 0) {

                this.state.offset(outputs*batch);
            }
            if(this.shortcut != 0){
                oldState.copyInto(outputs*batch,this.state);
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

            inputLayer.output.copyInto(outputs*batch,this.state);
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

            selfLayer.delta.copyInto(outputs*batch,inputLayer.delta);
            if (i > 0 && this.shortcut != 0) {

                FloatArray fb = selfLayer.delta.offsetNew(-outputs*batch);
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
