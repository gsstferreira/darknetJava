package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Buffers.IntBuffer;
import Classes.Image;
import Classes.Layer;
import Classes.Network;
import Yolo.Enums.LayerType;
import Tools.Buffers;
import Tools.Rand;


public class MaxpoolLayer extends Layer {

    public Image getMaxpoolImage() {

        int h = outH;
        int w = outW;

        return new Image(w,h,c,output);
    }

    public Image getMaxpoolDelta() {

        int h = outH;
        int w = outW;

        return new Image(w,h,c,delta);
    }

    public MaxpoolLayer(int batch, int height, int width, int c, int size, int stride, int padding) {

        type = LayerType.MAXPOOL;
        this.batch = batch;
        h = height;
        w = width;
        this.c = c;
        pad = padding;
        outW = (width + padding - size)/stride + 1;
        outH = (height + padding - size)/stride + 1;
        outC = c;
        outputs = outH * outW * outC;
        inputs = height*width*c;
        this.size = size;
        this.stride = stride;
        int output_size = outH * outW * outC * batch;
        indexes = new IntBuffer(output_size);
        output = new FloatBuffer(output_size);
        delta =  new FloatBuffer(output_size);

        System.out.printf("max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, outW, outH, outC);
    }

    public void resize(int width, int height) {

        h = height;
        w = width;
        inputs = height*width*c;

        outW = (width + pad - size)/stride + 1;
        outH = (height + pad - size)/stride + 1;
        outputs = outW * outH * c;
        int output_size = outputs * batch;

        indexes = Buffers.realloc(indexes,output_size);
        output = Buffers.realloc(output,output_size);
        delta = Buffers.realloc(delta,output_size);
    }

    public void forward(Network net) {

        int b,i,j,k,m,n;
        int w_offset = -pad/2;
        int h_offset = -pad/2;

        int h = outH;
        int w = outW;
        int c = this.c;

        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                for(i = 0; i < h; ++i){
                    for(j = 0; j < w; ++j){
                        int out_index = j + w*(i + h*(k + c*b));
                        float max = -Rand.MAX_FLOAT;
                        int max_i = -1;
                        for(n = 0; n < size; ++n){
                            for(m = 0; m < size; ++m){
                                int cur_h = h_offset + i*stride + n;
                                int cur_w = w_offset + j*stride + m;
                                int index = cur_w + this.w*(cur_h + this.h*(k + b*this.c));
                                boolean valid = (cur_h >= 0 && cur_h < this.h && cur_w >= 0 && cur_w < this.w);

                                float val = (valid) ? net.input.get(index) : -Rand.MAX_FLOAT;

                                max_i = (val > max) ? index : max_i;
                                max   = (val > max) ? val   : max;
                            }
                        }
                        output.put(out_index,max);
                        indexes.put(out_index,max_i);
                    }
                }
            }
        }
    }

    public void backward(Network net) {

        int i;
        int h = outH;
        int w = outW;
        int c = this.c;

        for(i = 0; i < h*w*c*batch; ++i){

            int index = indexes.get(i);
            net.delta.put(index,net.delta.get(index) + delta.get(i));
        }
    }
}
