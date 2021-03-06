package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Buffers;
import Tools.Util;
import Yolo.Enums.CostType;
import Yolo.Enums.LayerType;

public class CostLayer extends Layer {

    public CostLayer(int batch, int inputs, CostType costType, float scale) {

        this.type = LayerType.COST;

        this.scale = scale;
        this.batch = batch;
        this.inputs = inputs;
        this.outputs = inputs;
        this.costType = costType;

        this.delta = new FloatArray(inputs*batch);
        this.output = new FloatArray(inputs*batch);
        this.cost = new FloatArray(1);
    }

    public void resize(int inputs) {

        this.inputs = inputs;
        this.outputs = inputs;
        this.delta = Buffers.realloc(delta,inputs*batch);
        this.output = Buffers.realloc(output,inputs*batch);
    }

    public void forward(Network net) {

        if (net.truth != null) {
            if(costType == CostType.MASKED){
                int i;
                for(i = 0; i < batch*inputs; ++i){
                    if(net.truth.get(i) == Util.SECRET_NUMBER) {
                        net.input.set(i,Util.SECRET_NUMBER);
                    }
                }
            }
            else if(costType == CostType.SMOOTH){
                Blas.smoothL1Cpu(batch * inputs, net.input, net.truth, delta, output);
            }
            else if(costType == CostType.L1){
                Blas.l1Cpu(batch*inputs, net.input, net.truth, delta, output);
            }
            else {
                Blas.l2Cpu(batch*inputs, net.input, net.truth, delta, output);
            }
            cost.set(0,Util.sumArray(output,batch*inputs));
        }
    }

    public void backward(Network net) {

        Blas.axpyCpu(batch*inputs, scale, delta, 1, net.delta, 1);
    }
}
