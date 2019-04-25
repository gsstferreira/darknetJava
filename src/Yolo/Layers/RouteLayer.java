package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Buffers;
import Yolo.Enums.LayerType;



public class RouteLayer extends Layer {

    public RouteLayer(int batch, int n, IntArray inputLayers, IntArray inputSizes) {

        StringBuilder sb = new StringBuilder();
        sb.append("Route ");

        this.type = LayerType.ROUTE;
        this.batch = batch;
        this.n = n;
        this.inputLayers = inputLayers;
        this.inputSizes = inputSizes;
        int i;
        int outputs = 0;

        for(i = 0; i < n; ++i){

            sb.append(String.format("%4d ", this.inputLayers.get(i)));
            outputs += inputSizes.get(i);
        }
        System.out.println(sb.toString());

        this.outputs = outputs;
        this.inputs = outputs;
        this.delta = new FloatArray(outputs * batch);
        this.output = new FloatArray(outputs * batch);
    }

    public void resize(Network net) {

        int i;
        Layer first = net.layers[this.inputLayers.get(0)];
        this.outW = first.outW;
        this.outH = first.outH;
        this.outC = first.outC;
        this.outputs = first.outputs;
        inputSizes.set(0,first.outputs);

        for(i = 1; i < this.n; ++i){

            int index = this.inputLayers.get(i);
            Layer next = net.layers[index];

            this.outputs += next.outputs;
            this.inputSizes.set(i,next.outputs);

            if(next.outW == first.outW && next.outH == first.outH){
                this.outC += next.outC;
            }
            else{
                this.outH = this.outW = this.outC = 0;
            }
        }
        this.inputs = this.outputs;
        this.delta =  Buffers.realloc(this.delta, this.outputs*this.batch);
        this.output = Buffers.realloc(this.output, this.outputs*this.batch);
    }

    public void forward(Network net) {

        int offset = 0;
        for(int i = 0; i < this.n; ++i){
            final int index = this.inputLayers.get(i);
            final int inputSize = this.inputSizes.get(i);

            final FloatArray fb1 = net.layers[index].output.shallowClone();
            final FloatArray fb2 = this.output.offsetNew(offset);

            for(int j = 0; j < this.batch; ++j){

                fb1.copyInto(inputSize,fb2);
                fb1.offset(inputSize);
                fb2.offset(this.outputs);
            }
            offset += inputSize;
        }
    }

    public void backward(Network net) {

        int i, j;
        int offset = 0;
        for(i = 0; i < this.n; ++i){
            int index = this.inputLayers.get(i);
            FloatArray delta = net.layers[index].delta;

            int input_size = this.inputSizes.get(i);

            for(j = 0; j < this.batch; ++j){

                FloatArray fb1 = this.delta.offsetNew(offset + j*this.outputs);
                FloatArray fb2 = delta.offsetNew(j*input_size);

                Blas.axpyCpu(input_size, 1, fb1, 1, fb2, 1);
            }
            offset += input_size;
        }
    }
}
