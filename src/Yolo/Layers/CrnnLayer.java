package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Tools.Blas;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

public class CrnnLayer extends Layer {

    public static void incrementLayer(Layer l, int steps) {

        int num = l.outputs*l.batch*steps;

        l.output.offset(num);
        l.delta.offset(num);
        l.x.offset(num);
        l.xNorm.offset(num);
    }

    public CrnnLayer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, Activation activation, int batch_normalize) {

        batch = batch / steps;

        this.batch = batch;
        this.type = LayerType.CRNN;
        this.steps = steps;
        this.h = h;
        this.w = w;
        this.c = c;
        this.outH = h;
        this.outW = w;
        this.outC = output_filters;
        this.inputs = h*w*c;
        this.hidden = h * w * hidden_filters;
        this.outputs = this.outH * this.outW * this.outC;

        this.state = new FloatArray(this.hidden*batch*(steps + 1));

        this.inputLayer = new ConvolutionalLayer(batch*steps, h, w, c, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
        this.inputLayer.batch = batch;

        this.selfLayer = new ConvolutionalLayer(batch*steps, h, w, hidden_filters, hidden_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
        this.selfLayer.batch = batch;

        this.outputLayer = new ConvolutionalLayer(batch*steps, h, w, hidden_filters, output_filters, 1, 3, 1, 1,  activation, batch_normalize, 0, 0, 0);
        this.outputLayer.batch = batch;

        this.output = this.outputLayer.output;
        this.delta = this.outputLayer.delta;

    }

    public void update(UpdateArgs a) {

        ((ConvolutionalLayer)inputLayer).update(a);
        ((ConvolutionalLayer)selfLayer).update(a);
        ((ConvolutionalLayer)outputLayer).update(a);
    }

    public void forward(Network net) {

        try {
            Network s = (Network) net.clone();
            s.train = net.train;
            int i;

            Layer input_Layer = inputLayer;
            Layer self_Layer =  selfLayer;
            Layer output_Layer = outputLayer;

            Blas.fillCpu(outputs * batch * steps, 0, output_Layer.delta, 1);
            Blas.fillCpu(hidden * batch * steps, 0, self_Layer.delta, 1);
            Blas.fillCpu(hidden * batch * steps, 0, input_Layer.delta, 1);

            if(net.train != 0) {
                Blas.fillCpu(hidden * batch, 0, state, 1);
            }

            for (i = 0; i < steps; ++i) {

                s.input = net.input;
                ((ConvolutionalLayer)input_Layer).forward(s);

                s.input = state;
                ((ConvolutionalLayer)self_Layer).forward(s);

                FloatArray oldState = state;

                if(net.train != 0) {
                    state.offset(hidden*batch);
                }
                if(shortcut != 0){

                    oldState.copyInto(hidden*batch,state);
                }
                else{
                    Blas.fillCpu(hidden *batch, 0, state, 1);
                }

                Blas.axpyCpu(hidden * batch, 1, input_Layer.output, 1, state, 1);
                Blas.axpyCpu(hidden * batch, 1, self_Layer.output, 1, state, 1);

                s.input = state;
                ((ConvolutionalLayer)output_Layer).forward(s);

                net.input.offset(inputs*batch);

                incrementLayer(input_Layer, 1);
                incrementLayer(self_Layer, 1);
                incrementLayer(output_Layer, 1);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public void backward(Network net) {

        try {
            Network s = net.clone();
            int i;

            incrementLayer(inputLayer, steps-1);
            incrementLayer(selfLayer, steps-1);
            incrementLayer(outputLayer, steps-1);

            state.offset(hidden*batch*steps);

            for (i = steps - 1; i >= 0; --i) {

                inputLayer.output.copyInto(hidden*batch,state);
                Blas.axpyCpu(hidden * batch, 1, selfLayer.output, 1, state, 1);

                s.input = state;
                s.delta = selfLayer.delta;

                ((ConvolutionalLayer)outputLayer).backward(s);

                state.offset(-hidden*batch);

                s.input = state;
                delta = selfLayer.delta.offsetNew(-hidden*batch);

                if (i == 0) {
                    s.delta = null;
                }

                ((ConvolutionalLayer)selfLayer).backward(s);

                selfLayer.delta.copyInto(hidden*batch,inputLayer.delta);

                if (i > 0 && shortcut != 0) {

                    FloatArray fb = selfLayer.delta.offsetNew(-hidden*batch);
                    Blas.axpyCpu(hidden*batch, 1, selfLayer.delta, 1, fb, 1);
                }

                s.input = net.input.offsetNew(i*inputs*batch);

                if(net.delta != null) {

                    s.delta = net.delta.offsetNew(i*inputs*batch);
                }
                else {
                    s.delta = null;
                }

                ((ConvolutionalLayer)inputLayer).backward(s);

                incrementLayer(inputLayer, -1);
                incrementLayer(selfLayer, -1);
                incrementLayer(outputLayer, -1);
            }
        }
        catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

}
