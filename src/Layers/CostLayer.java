package Layers;

import Classes.Layer;
import Classes.Network;
import Enums.CostType;
import Enums.LayerType;
import Tools.Blas;
import Tools.BufferUtil;
import Tools.Util;
import org.lwjgl.BufferUtils;

public class CostLayer extends Layer {

    public CostLayer(int batch, int inputs, CostType costType, float scale) {

        this.type = LayerType.COST;

        this.scale = scale;
        this.batch = batch;
        this.inputs = inputs;
        this.outputs = inputs;
        this.costType = costType;

        this.delta = BufferUtils.createFloatBuffer(inputs*batch);
        this.output = BufferUtils.createFloatBuffer(inputs*batch);
        this.cost = BufferUtils.createFloatBuffer(1);
    }

    public void resize(int inputs) {

        this.inputs = inputs;
        this.outputs = inputs;
        this.delta = BufferUtil.reallocBuffer(delta,inputs*batch);
        this.output = BufferUtil.reallocBuffer(output,inputs*batch);
    }

    public void forward(Network net) {

        if (net.truth != null) {
            if(costType == CostType.MASKED){
                int i;
                for(i = 0; i < batch*inputs; ++i){
                    if(net.truth.get(i) == Util.SECRET_NUMBER) {
                        net.input.put(i,Util.SECRET_NUMBER);
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
            cost.put(0,Util.sumArray(output,batch*inputs));
        }
    }

    public void backward(Network net) {

        Blas.axpyCpu(batch*inputs, scale, delta, 1, net.delta, 1);
    }
}