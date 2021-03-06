package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Tools.Buffers;
import Tools.Rand;
import Yolo.Enums.LayerType;

public class DropoutLayer extends Layer {

    public DropoutLayer(int batch, int inputs, float probability) {
        
        this.type = LayerType.DROPOUT;
        this.probability = probability;
        this.inputs = inputs;
        this.outputs = inputs;
        this.batch = batch;
        this.rand = new FloatArray(inputs*batch);
        this.scale = 1.0f/(1.0f - probability);        
    }

    public void resize(int inputs) {
        
        rand = Buffers.realloc(rand,inputs*batch);
    }

    public void forward(Network net) {

        int i;
        if (net.train != 0) {
            return;
        }
        for(i = 0; i < batch * inputs; ++i){

            float r = Rand.randUniform(0, 1);
            this.rand.set(i,r);

            if(r < probability) {
                net.input.set(i,0);
            }
            else {
                net.input.set(i,net.input.get(i) * scale);
            }
        }
    }

    public void backward(Network net) {

        int i;
        if(net.delta == null) {
            return;
        }

        for(i = 0; i < batch * inputs; ++i){
            float r = rand.get(i);

            if(r < this.probability) {
                net.delta.set(i,0);
            }
            else {
                net.delta.set(i,net.delta.get(i) * scale);
            }
        }
    }
}
