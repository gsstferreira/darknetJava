package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Util;
import Yolo.Enums.LayerType;

public class SoftmaxLayer extends Layer {

    public SoftmaxLayer(int batch, int inputs, int groups) {

        assert(inputs%groups == 0);
        
        this.type = LayerType.SOFTMAX;
        this.batch = batch;
        this.groups = groups;
        this.inputs = inputs;
        this.outputs = inputs;
        this.loss = new FloatBuffer(inputs*batch);
        this.output = new FloatBuffer(inputs*batch);
        this.delta = new FloatBuffer(inputs*batch);
        this.cost = new FloatBuffer(1);

        System.out.printf("softmax                                        %4d\n",  inputs);
    }

    public void forward(Network net) {

        if(this.softmaxTree != null){
            int i;
            int count = 0;
            for (i = 0; i < this.softmaxTree.groups; ++i) {
                int group_size = this.softmaxTree.groupSize[i];

                FloatBuffer fb1 = net.input.offsetNew(count);
                FloatBuffer fb2 = net.output.offsetNew(count);

                Blas.softmaxCpu(fb1, group_size, this.batch, this.inputs, 1, 0, 1, this.temperature, fb2);
                count += group_size;
            }
        } else {
            Blas.softmaxCpu(net.input, this.inputs/this.groups, this.batch, this.inputs, this.groups, this.inputs/this.groups, 1, this.temperature, this.output);
        }

        if(net.truth != null && this.noloss == 0){
            Blas.softmaxXEntCpu(this.batch*this.inputs, this.output, net.truth, this.delta, this.loss);

            this.cost.put(0,Util.sumArray(this.loss, this.batch * this.inputs));
        }
    }

    public void backward(Network net) {

        Blas.axpyCpu(this.inputs*this.batch, 1, this.delta, 1, net.delta, 1);
    }

}
