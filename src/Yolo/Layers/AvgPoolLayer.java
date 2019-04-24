package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Layer;
import Classes.Network;
import Yolo.Enums.LayerType;

public class AvgPoolLayer extends Layer {

    public AvgPoolLayer(int batch, int w, int h, int c) {

        type = LayerType.AVGPOOL;
        this.batch = batch;
        this.h = h;
        this.w = w;
        this.c = c;
        this.outW = 1;
        this.outH = 1;
        this.outC = c;
        this.outputs = this.outC;
        this.inputs = h*w*c;

        int output_size = this.outputs * batch;
        this.output = new FloatArray(output_size);
        this.delta =  new FloatArray(output_size);
    }

    public void resize(int width, int height) {

        w = width;
        h = height;
        inputs = height*width*c;
    }

    public void forward(Network net) {

        int b,i,k;

        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                int out_index = k + b*c;
                output.put(out_index,0);
                for(i = 0; i < h*w; ++i){

                    int in_index = i + h*w*(k + b*c);
                    output.put(out_index,output.get(out_index) + net.input.get(in_index));
                }
                output.put(out_index,output.get(out_index) / (h*w));
            }
        }
    }

    public void backward(Network net) {

        int b,i,k;

        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                int out_index = k + b*c;
                for(i = 0; i < h*w; ++i){
                    int in_index = i + h*w*(k + b*c);

                    float val = net.delta.get(in_index) + this.delta.get(out_index)/(h*w);
                    net.delta.put(in_index,val);
                }
            }
        }
    }
}
