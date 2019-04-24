package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Tools.Blas;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;

public class LstmLayer extends Layer {

    public static void incrementLayer(Layer l, int steps) {

        int num = l.outputs*l.batch*steps;

        l.output.offset(num);
        l.delta.offset(num);
        l.x.offset(num);
        l.xNorm.offset(num);
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

        this.output = new FloatArray(outputs*batch*steps);
        this.state = new FloatArray(outputs*batch);
        
        this.prevStateCpu = new FloatArray(batch*outputs);
        this.prevCellCpu = new FloatArray(batch*outputs);
        this.cellCpu = new FloatArray(batch*outputs*steps);

        this.fCpu = new FloatArray(batch*outputs);
        this.iCpu = new FloatArray(batch*outputs);
        this.gCpu = new FloatArray(batch*outputs);
        this.oCpu = new FloatArray(batch*outputs);
        this.cCpu = new FloatArray(batch*outputs);
        this.hCpu = new FloatArray(batch*outputs);
        this.tempCpu = new FloatArray(batch*outputs);
        this.temp2Cpu = new FloatArray(batch*outputs);
        this.temp3Cpu = new FloatArray(batch*outputs);
        this.dcCpu = new FloatArray(batch*outputs);
        this.dhCpu = new FloatArray(batch*outputs);
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

            wf.output.copyInto(outputs*batch,fCpu);
            Blas.axpyCpu(outputs*batch, 1, uf.output, 1, fCpu, 1);

            wi.output.copyInto(outputs*batch,iCpu);
            Blas.axpyCpu(outputs*batch, 1, ui.output, 1, iCpu, 1);

            wg.output.copyInto(outputs*batch,gCpu);
            Blas.axpyCpu(outputs*batch, 1, ug.output, 1, gCpu, 1);

            wo.output.copyInto(outputs*batch,oCpu);
            Blas.axpyCpu(outputs*batch, 1, uo.output, 1, oCpu, 1);

            Activation.activateArray(fCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(iCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(gCpu, outputs*batch, Activation.TANH);
            Activation.activateArray(oCpu, outputs*batch, Activation.LOGISTIC);

            iCpu.copyInto(outputs*batch,tempCpu);
            Blas.mulCpu(outputs*batch, gCpu, 1, tempCpu, 1);
            Blas.mulCpu(outputs*batch, fCpu, 1, cCpu, 1);
            Blas.axpyCpu(outputs*batch, 1, tempCpu, 1, cCpu, 1);

            cCpu.copyInto(outputs*batch,hCpu);
            Activation.activateArray(hCpu, outputs*batch, Activation.TANH);
            Blas.mulCpu(outputs*batch, oCpu, 1, hCpu, 1);

            cCpu.copyInto(outputs*batch,cellCpu);
            hCpu.copyInto(outputs*batch,output);

            state.input.offset(inputs*batch);
            output.offset(outputs*batch);
            cellCpu.offset(outputs*batch);

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

        state.input.offset(inputs*batch*(steps - 1));

        if (state.delta != null) {

            state.delta.offset(inputs*batch*(steps - 1));
        }

        output.offset(outputs*batch*(steps - 1));
        cellCpu.offset(outputs*batch*(steps - 1));
        delta.offset(outputs*batch*(steps - 1));
        
        for (i = steps - 1; i >= 0; --i) {

            FloatArray fba = cellCpu.offsetNew(-outputs*batch);
            FloatArray fbb = output.offsetNew(-outputs*batch);
            
            if (i != 0) {
                fba.copyInto(outputs*batch,prevCellCpu);
            }
            cellCpu.copyInto(outputs*batch,cCpu);
            
            if (i != 0) {
                fbb.copyInto(outputs*batch,prevStateCpu);
            }
            output.copyInto(outputs*batch,hCpu);

            dhCpu = (i == 0) ? null : delta.offsetNew(-outputs*batch);

            wf.output.copyInto(outputs*batch,fCpu);
            Blas.axpyCpu(outputs*batch, 1, uf.output, 1, fCpu, 1);

            wi.output.copyInto(outputs*batch,iCpu);
            Blas.axpyCpu(outputs*batch, 1, ui.output, 1, iCpu, 1);

            wg.output.copyInto(outputs*batch,gCpu);
            Blas.axpyCpu(outputs*batch, 1, ug.output, 1, gCpu, 1);

            wo.output.copyInto(outputs*batch,oCpu);
            Blas.axpyCpu(outputs*batch, 1, uo.output, 1, oCpu, 1);

            Activation.activateArray(fCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(iCpu, outputs*batch, Activation.LOGISTIC);
            Activation.activateArray(gCpu, outputs*batch, Activation.TANH);
            Activation.activateArray(oCpu, outputs*batch, Activation.LOGISTIC);

            delta.copyInto(outputs*batch,temp3Cpu);

            cCpu.copyInto(outputs*batch,tempCpu);
            Activation.activateArray(tempCpu, outputs*batch, Activation.TANH);

            temp3Cpu.copyInto(outputs*batch,temp2Cpu);
            Blas.mulCpu(outputs*batch, oCpu, 1, temp2Cpu, 1);

            Activation.gradientArray(tempCpu, outputs*batch, Activation.TANH, temp2Cpu);
            Blas.axpyCpu(outputs*batch, 1, dcCpu, 1, temp2Cpu, 1);

            cCpu.copyInto(outputs*batch,tempCpu);
            Activation.activateArray(tempCpu, outputs*batch, Activation.TANH);
            Blas.mulCpu(outputs*batch, temp3Cpu, 1, tempCpu, 1);
            Activation.gradientArray(oCpu, outputs*batch, Activation.LOGISTIC, tempCpu);
            tempCpu.copyInto(outputs*batch,wo.delta);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wo).backward(s);

            tempCpu.copyInto(outputs*batch,uo.delta);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)uo).backward(s);

            temp2Cpu.copyInto(outputs*batch,tempCpu);
            Blas.mulCpu(outputs*batch, iCpu, 1, tempCpu, 1);
            Activation.gradientArray(gCpu, outputs*batch, Activation.TANH, tempCpu);
            tempCpu.copyInto(outputs*batch,wg.delta);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wg).backward(s);

            tempCpu.copyInto(outputs*batch,ug.delta);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)ug).backward(s);

            temp2Cpu.copyInto(outputs*batch,tempCpu);
            Blas.mulCpu(outputs*batch, gCpu, 1, tempCpu, 1);
            Activation.gradientArray(iCpu, outputs*batch, Activation.LOGISTIC, tempCpu);
            tempCpu.copyInto(outputs*batch,wi.delta);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wi).backward(s);

            tempCpu.copyInto(outputs*batch,ui.delta);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)ui).backward(s);

            temp2Cpu.copyInto(outputs*batch,tempCpu);
            Blas.mulCpu(outputs*batch, prevCellCpu, 1, tempCpu, 1);
            Activation.gradientArray(fCpu, outputs*batch, Activation.LOGISTIC, tempCpu);
            tempCpu.copyInto(outputs*batch,wf.delta);
            s.input = prevStateCpu;
            s.delta = dhCpu;

            ((ConnectedLayer)wf).backward(s);

            tempCpu.copyInto(outputs*batch,uf.delta);
            s.input = state.input;
            s.delta = state.delta;

            ((ConnectedLayer)uf).backward(s);

            temp2Cpu.copyInto(outputs*batch,tempCpu);
            Blas.mulCpu(outputs*batch, fCpu, 1, tempCpu, 1);
            tempCpu.copyInto(outputs*batch,dcCpu);

            state.input.offset(-inputs*batch);

            if (state.delta != null) {

                state.delta.offset(-inputs*batch);
            }

            output.offset(-outputs*batch);
            cellCpu.offset(-outputs*batch);
            delta.offset(-outputs*batch);

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
