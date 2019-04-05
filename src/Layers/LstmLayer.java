package Layers;

import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Enums.Activation;
import Enums.LayerType;
import Tools.Blas;
import Tools.Buffers;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

public class LstmLayer extends Layer {

    public static void incrementLayer(Layer l, int steps) {

        int num = l.outputs*l.batch*steps;

        l.output = Buffers.offset(l.output,num);
        l.delta = Buffers.offset(l.delta,num);
        l.x = Buffers.offset(l.x,num);;
        l.xNorm = Buffers.offset(l.xNorm,num);
    }

    public LstmLayer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam) {

        batch = batch / steps;

        this.batch = batch;
        this.type = LayerType.LSTM;
        this.steps = steps;
        this.inputs = inputs;

        this.uf = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.uf.batch = batch;

        this.ui = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.ui.batch = batch;

        this.ug = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.ug.batch = batch;

        this.uo = new ConnectedLayer(batch*steps, inputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.uo.batch = batch;

        this.wf = new ConnectedLayer(batch*steps, outputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.wf.batch = batch;

        this.wi = new ConnectedLayer(batch*steps, outputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.wi.batch = batch;

        this.wg = new ConnectedLayer(batch*steps, outputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.wg.batch = batch;

        this.wo = new ConnectedLayer(batch*steps, outputs, outputs, Activation.LINEAR, batch_normalize, adam);
        this.wo.batch = batch;

        this.batchNormalize = batch_normalize;
        this.outputs = outputs;

        this.output = Buffers.newBufferF(outputs*batch*steps);
        this.state = Buffers.newBufferF(outputs*batch);
        
        this.prevStateCpu = Buffers.newBufferF(batch*outputs);
        this.prevCellCpu = Buffers.newBufferF(batch*outputs);
        this.cellCpu = Buffers.newBufferF(batch*outputs*steps);

        this.fCpu = Buffers.newBufferF(batch*outputs);
        this.iCpu = Buffers.newBufferF(batch*outputs);
        this.gCpu = Buffers.newBufferF(batch*outputs);
        this.oCpu = Buffers.newBufferF(batch*outputs);
        this.cCpu = Buffers.newBufferF(batch*outputs);
        this.hCpu = Buffers.newBufferF(batch*outputs);
        this.tempCpu = Buffers.newBufferF(batch*outputs);
        this.temp2Cpu = Buffers.newBufferF(batch*outputs);
        this.temp3Cpu = Buffers.newBufferF(batch*outputs);
        this.dcCpu = Buffers.newBufferF(batch*outputs);
        this.dhCpu = Buffers.newBufferF(batch*outputs);
    }

    public void update(UpdateArgs a) {

        ((ConnectedLayer)wf).update(a);
        ((ConnectedLayer)wi).update(a);
        ((ConnectedLayer)wg).update(a);
        ((ConnectedLayer)wo).update(a);
        ((ConnectedLayer)uf).update(a);
        ((ConnectedLayer)ui).update(a);
        ((ConnectedLayer)ug).update(a);
        ((ConnectedLayer)uo).update(a);
    }

    public void forward(Network state) {
        
        Network s = new Network();
        s.train = state.train;
        int i;

        Blas.fillCpu(outputs * batch * steps, 0, wf.delta, 1);
        Blas.fillCpu(outputs * batch * steps, 0, wi.delta, 1);
        Blas.fillCpu(outputs * batch * steps, 0, wg.delta, 1);
        Blas.fillCpu(outputs * batch * steps, 0, wo.delta, 1);

        Blas.fillCpu(outputs * batch * steps, 0, uf.delta, 1);
        Blas.fillCpu(outputs * batch * steps, 0, ui.delta, 1);
        Blas.fillCpu(outputs * batch * steps, 0, ug.delta, 1);
        Blas.fillCpu(outputs * batch * steps, 0, uo.delta, 1);

        if (state.train != 0) {
            Blas.fillCpu(outputs * batch * steps, 0, delta, 1);
        }

        for (i = 0; i < steps; ++i) {
            s.input = hCpu;
            
            ((ConnectedLayer)wf).forward(s);
            ((ConnectedLayer)wi).forward(s);
            ((ConnectedLayer)wg).forward(s);
            ((ConnectedLayer)wo).forward(s);

            s.input = state.input;
            
            ((ConnectedLayer)uf).forward(s);
            ((ConnectedLayer)ui).forward(s);
            ((ConnectedLayer)ug).forward(s);
            ((ConnectedLayer)uo).forward(s);

            Blas.copyCpu(outputs*batch, wf.output, 1, fCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, uf.output, 1, fCpu, 1);

            Blas.copyCpu(outputs*batch, wi.output, 1, iCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, ui.output, 1, iCpu, 1);

            Blas.copyCpu(outputs*batch, wg.output, 1, gCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, ug.output, 1, gCpu, 1);

            Blas.copyCpu(outputs*batch, wo.output, 1, oCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, uo.output, 1, oCpu, 1);

            Activation.activateArray(fCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(iCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(gCpu, outputs*batch, Activation.TANH);
            Activation.activateArray(oCpu, outputs*batch, Activation.LOGISTIC);

            Blas.copyCpu(outputs*batch, iCpu, 1, tempCpu, 1);
            Blas.mulCpu(outputs*batch, gCpu, 1, tempCpu, 1);
            Blas.mulCpu(outputs*batch, fCpu, 1, cCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, tempCpu, 1, cCpu, 1);

            Blas.copyCpu(outputs*batch, cCpu, 1, hCpu, 1);
            Activation.activateArray(hCpu, outputs*batch, Activation.TANH);
            Blas.mulCpu(outputs*batch, oCpu, 1, hCpu, 1);

            Blas.copyCpu(outputs*batch, cCpu, 1, cellCpu, 1);
            Blas.copyCpu(outputs*batch, hCpu, 1, output, 1);

            state.input = Buffers.offset(state.input,inputs*batch);
            output = Buffers.offset(output,outputs*batch);
            cellCpu = Buffers.offset(cellCpu,outputs*batch);

            incrementLayer(wf, 1);
            incrementLayer(wi, 1);
            incrementLayer(wg, 1);
            incrementLayer(wo, 1);

            incrementLayer(uf, 1);
            incrementLayer(ui, 1);
            incrementLayer(ug, 1);
            incrementLayer(uo, 1);
        }
    }

    public void backward(Network state) {
        
        Network s = new Network();
        s.train = state.train;
        int i;

        incrementLayer(wf, steps - 1);
        incrementLayer(wi, steps - 1);
        incrementLayer(wg, steps - 1);
        incrementLayer(wo, steps - 1);

        incrementLayer(uf, steps - 1);
        incrementLayer(ui, steps - 1);
        incrementLayer(ug, steps - 1);
        incrementLayer(uo, steps - 1);

        state.input = Buffers.offset(state.input,inputs*batch*(steps - 1));

        if (state.delta != null) {

            state.delta = Buffers.offset(state.delta,inputs*batch*(steps - 1));
        }

        output = Buffers.offset(output,outputs*batch*(steps - 1));
        cellCpu = Buffers.offset(cellCpu,outputs*batch*(steps - 1));
        delta = Buffers.offset(delta,outputs*batch*(steps - 1));
        
        for (i = steps - 1; i >= 0; --i) {

            FloatBuffer fba = Buffers.offset(cellCpu, -outputs*batch);
            FloatBuffer fbb = Buffers.offset(output, -outputs*batch);
            
            if (i != 0) {
                Blas.copyCpu(outputs*batch, fba, 1, prevCellCpu, 1);
            }
            Blas.copyCpu(outputs*batch, cellCpu, 1,cCpu, 1);
            
            if (i != 0) {
                Blas.copyCpu(outputs*batch, fbb, 1, prevStateCpu, 1);
            }
            Blas.copyCpu(outputs*batch, output, 1, hCpu, 1);

            dhCpu = (i == 0) ? null : Buffers.offset(delta, - outputs*batch);

            Blas.copyCpu(outputs*batch, wf.output, 1, fCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, uf.output, 1, fCpu, 1);

            Blas.copyCpu(outputs*batch, wi.output, 1, iCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, ui.output, 1, iCpu, 1);

            Blas.copyCpu(outputs*batch, wg.output, 1, gCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, ug.output, 1, gCpu, 1);

            Blas.copyCpu(outputs*batch, wo.output, 1, oCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, uo.output, 1, oCpu, 1);

            Activation.activateArray(fCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(iCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(gCpu, outputs*batch, Activation.TANH);
            Activation.activateArray(oCpu, outputs*batch, Activation.LOGISTIC);

            Blas.copyCpu(outputs*batch, delta, 1, temp3Cpu, 1);

            Blas.copyCpu(outputs*batch,cCpu, 1, tempCpu, 1);
            Activation.activateArray(tempCpu, outputs*batch, Activation.TANH);

            Blas.copyCpu(outputs*batch, temp3Cpu, 1, temp2Cpu, 1);
            Blas.mulCpu(outputs*batch, oCpu, 1, temp2Cpu, 1);

            Activation.gradientArray(tempCpu, outputs*batch, Activation.TANH, temp2Cpu);
            Blas.axpyCpu(outputs*batch, 1, dcCpu, 1, temp2Cpu, 1);

            Blas.copyCpu(outputs*batch,cCpu, 1, tempCpu, 1);
            Activation.activateArray(tempCpu, outputs*batch, Activation.TANH);
            Blas.mulCpu(outputs*batch, temp3Cpu, 1, tempCpu, 1);
            Activation.gradientArray(oCpu, outputs*batch, Activation.LOGISTIC, tempCpu);
            Blas.copyCpu(outputs*batch, tempCpu, 1, wo.delta, 1);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wo).backward(s);

            Blas.copyCpu(outputs*batch, tempCpu, 1, uo.delta, 1);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)uo).backward(s);

            Blas.copyCpu(outputs*batch, temp2Cpu, 1, tempCpu, 1);
            Blas.mulCpu(outputs*batch, iCpu, 1, tempCpu, 1);
            Activation.gradientArray(gCpu, outputs*batch, Activation.TANH, tempCpu);
            Blas.copyCpu(outputs*batch, tempCpu, 1, wg.delta, 1);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wg).backward(s);

            Blas.copyCpu(outputs*batch, tempCpu, 1, ug.delta, 1);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)ug).backward(s);

            Blas.copyCpu(outputs*batch, temp2Cpu, 1, tempCpu, 1);
            Blas.mulCpu(outputs*batch, gCpu, 1, tempCpu, 1);
            Activation.gradientArray(iCpu, outputs*batch, Activation.LOGISTIC, tempCpu);
            Blas.copyCpu(outputs*batch, tempCpu, 1, wi.delta, 1);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wi).backward(s);

            Blas.copyCpu(outputs*batch, tempCpu, 1, ui.delta, 1);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)ui).backward(s);

            Blas.copyCpu(outputs*batch, temp2Cpu, 1, tempCpu, 1);
            Blas.mulCpu(outputs*batch, prevCellCpu, 1, tempCpu, 1);
            Activation.gradientArray(fCpu, outputs*batch, Activation.LOGISTIC, tempCpu);
            Blas.copyCpu(outputs*batch, tempCpu, 1, wf.delta, 1);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wf).backward(s);

            Blas.copyCpu(outputs*batch, tempCpu, 1, uf.delta, 1);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)uf).backward(s);

            Blas.copyCpu(outputs*batch, temp2Cpu, 1, tempCpu, 1);
            Blas.mulCpu(outputs*batch, fCpu, 1, tempCpu, 1);
            Blas.copyCpu(outputs*batch, tempCpu, 1, dcCpu, 1);

            state.input = Buffers.offset(state.input,-inputs*batch);

            if (state.delta != null) {

                state.delta = Buffers.offset(state.delta,-inputs*batch);
            }

            output = Buffers.offset(output,-outputs*batch);
            cellCpu = Buffers.offset(cellCpu,-outputs*batch);
            delta = Buffers.offset(delta,-outputs*batch);

            incrementLayer(wf, -1);
            incrementLayer(wi, -1);
            incrementLayer(wg, -1);
            incrementLayer(wo, -1);

            incrementLayer(uf, -1);
            incrementLayer(ui, -1);
            incrementLayer(ug, -1);
            incrementLayer(uo, -1);
        }
    }

}
