package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Yolo.Enums.LayerType;
import Tools.Buffers;
import Tools.Rand;

public class DropoutLayer extends Layer {

    public DropoutLayer(int batch, int inputs, float probability) {
        
        this.type = LayerType.DROPOUT;
        this.probability = probability;
        this.inputs = inputs;
        this.outputs = inputs;
        this.batch = batch;
        this.rand = new FloatBuffer(inputs*batch);
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
            this.rand.put(i,r);

            if(r < probability) {
                net.input.put(i,0);
            }
            else {
                net.input.put(i,net.input.get(i) * scale);
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
                net.delta.put(i,0);
            }
            else {
                net.delta.put(i,net.delta.get(i) * scale);
            }
        }
    }
}
