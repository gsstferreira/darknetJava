package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Tools.Blas;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

import static Yolo.Enums.Activation.LOGISTIC;

public class GruLayer extends Layer {

    public static void increment_Layer(Layer l, int steps) {

        int num = l.outputs*l.batch*steps;
        l.output.offset(num);
        l.delta.offset(num);
        l.x.offset(num);
        l.xNorm.offset(num);
        
    }

    public GruLayer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam) {

        batch = batch / steps;

        this.batch = batch;
        this.type = LayerType.GRU;
        this.steps = steps;
        this.inputs = inputs;

        this.uz = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.uz.batch = batch;

        this.wz = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.wz.batch = batch;

        this.ur = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.ur.batch = batch;

        this.wr = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.wr.batch = batch;

        this.uh = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.uh.batch = batch;

        this.wh = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.wh.batch = batch;
        
        this.batchNormalize = batch_normalize;
        
        this.outputs = outputs;
        this.output = new FloatArray(outputs*batch*steps);
        this.delta = new FloatArray(outputs*batch*steps);
        this.state = new FloatArray(outputs*batch);
        this.prevState = new FloatArray(outputs*batch);
        this.forgotState = new FloatArray(outputs*batch);
        this.forgotDelta = new FloatArray(outputs*batch);

        this.rCpu = new FloatArray(outputs*batch);
        this.zCpu = new FloatArray(outputs*batch);
        this.hCpu = new FloatArray(outputs*batch);
    }

    public void update(UpdateArgs a) {

        ((ConnectedLayer)ur).update(a);
        ((ConnectedLayer)uz).update(a);
        ((ConnectedLayer)uh).update(a);
        ((ConnectedLayer)wr).update(a);
        ((ConnectedLayer)wz).update(a);
        ((ConnectedLayer)wh).update(a);
    }

    public void forward(Network net) {

            Network s = net.tryClone();
            s.train = net.train;
            int i;

            Blas.fillCpu(outputs * batch * steps, 0, uz.delta, 1);
            Blas.fillCpu(outputs * batch * steps, 0, ur.delta, 1);
            Blas.fillCpu(outputs * batch * steps, 0, uh.delta, 1);

            Blas.fillCpu(outputs * batch * steps, 0, wz.delta, 1);
            Blas.fillCpu(outputs * batch * steps, 0, wr.delta, 1);
            Blas.fillCpu(outputs * batch * steps, 0, wh.delta, 1);

            if(net.train != 0) {
                Blas.fillCpu(outputs * batch * steps, 0, delta, 1);
                state.copyInto(outputs*batch,prevState);
            }

            for (i = 0; i < steps; ++i) {
                s.input = state;

                ((ConnectedLayer)wz).forward(s);
                ((ConnectedLayer)wr).forward(s);
                s.input = net.input;
                ((ConnectedLayer)uz).forward(s);
                ((ConnectedLayer)ur).forward(s);
                ((ConnectedLayer)uh).forward(s);

                uz.output.copyInto(outputs*batch,zCpu);
                Blas.axpyCpu(outputs*batch, 1, wz.output, 1, zCpu, 1);

                ur.output.copyInto(outputs*batch,rCpu);
                Blas.axpyCpu(outputs*batch, 1, wr.output, 1, rCpu, 1);

                Activation.activateArray(zCpu, outputs*batch, LOGISTIC);
                Activation.activateArray(rCpu, outputs*batch, LOGISTIC);

                state.copyInto(outputs*batch,forgotState);
                Blas.mulCpu(outputs*batch, rCpu, 1, forgotState, 1);

                s.input = forgotState;

                ((ConnectedLayer)wh).forward(s);

                uh.output.copyInto(outputs*batch,hCpu);
                Blas.axpyCpu(outputs*batch, 1, wh.output, 1, hCpu, 1);

                if(tanh != 0){
                    Activation.activateArray(hCpu, outputs*batch, Activation.TANH);
                } else {
                    Activation.activateArray(hCpu, outputs*batch, Activation.LOGISTIC);
                }

                Blas.weightedSumCpu(state, hCpu, zCpu, outputs*batch, output);
                output.copyInto(outputs*batch,state);

                net.input.offset(inputs*batch);
                output.offset(inputs*batch);

                increment_Layer(uz, 1);
                increment_Layer(ur, 1);
                increment_Layer(uh, 1);

                increment_Layer(wz, 1);
                increment_Layer(wr, 1);
                increment_Layer(wh, 1);
            }
    }

    @SuppressWarnings("EmptyMethod")
    public void backward(Network net){}

}
