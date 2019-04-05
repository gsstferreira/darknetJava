package Layers;

import Classes.Layer;
import Classes.Network;
import Enums.LayerType;
import Tools.Blas;
import Tools.Buffers;

import java.nio.IntBuffer;

public class RouteLayer extends Layer {

    public RouteLayer(int batch, int n, IntBuffer input_Layers, IntBuffer input_sizes) {

        this.type = LayerType.ROUTE;
        this.batch = batch;
        this.n = n;
        this.inputLayers = input_Layers;
        this.inputSizes = input_sizes;
        int i;
        int outputs = 0;
        for(i = 0; i < n; ++i){

            outputs += input_sizes.get(i);
        }

        this.outputs = outputs;
        this.inputs = outputs;
        this.delta = Buffers.newBufferF(outputs * batch);
        this.output = Buffers.newBufferF(outputs*batch);
    }

    public void resize(Network net) {

        int i;
        Layer first = net.layers[this.inputLayers.get(0)];
        this.outW = first.outW;
        this.outH = first.outH;
        this.outC = first.outC;
        this.outputs = first.outputs;
        inputSizes.put(0,first.outputs);

        for(i = 1; i < this.n; ++i){

            int index = this.inputLayers.get(i);
            Layer next = net.layers[index];

            this.outputs += next.outputs;
            this.inputSizes.put(i,next.outputs);

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

        int i, j;
        int offset = 0;
        for(i = 0; i < this.n; ++i){
            int index = this.inputLayers.get(i);
            var input = net.layers[index].output;
            int input_size = this.inputSizes.get(i);

            for(j = 0; j < this.batch; ++j){

                var fb1 = Buffers.offset(input,j*input_size);
                var fb2 = Buffers.offset(this.output,offset + j*this.outputs);

                Blas.copyCpu(input_size, fb1, 1, fb2, 1);
            }
            offset += input_size;
        }
    }

    public void backward(Network net) {

        int i, j;
        int offset = 0;
        for(i = 0; i < this.n; ++i){
            int index = this.inputLayers.get(i);
            var delta = net.layers[index].delta;

            int input_size = this.inputSizes.get(i);

            for(j = 0; j < this.batch; ++j){

                var fb1 = Buffers.offset(this.delta,offset + j*this.outputs);
                var fb2 = Buffers.offset(delta, j*input_size);

                Blas.axpyCpu(input_size, 1, fb1, 1, fb2, 1);
            }
            offset += input_size;
        }
    }
}
