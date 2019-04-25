package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Classes.Layer;
import Classes.Network;
import Tools.Buffers;
import Tools.Rand;
import Yolo.Enums.LayerType;

import java.util.stream.IntStream;


public class MaxpoolLayer extends Layer {

//    public Image getMaxpoolImage() {
//
//        int h = outH;
//        int w = outW;
//
//        return new Image(w,h,c,output);
//    }
//
//    public Image getMaxpoolDelta() {
//
//        int h = outH;
//        int w = outW;
//
//        return new Image(w,h,c,delta);
//    }

    public MaxpoolLayer(int batch, int height, int width, int c, int size, int stride, int padding) {

        this.type = LayerType.MAXPOOL;
        this.batch = batch;
        this.h = height;
        this.w = width;
        this.c = c;
        this.pad = padding;
        this.outW = (width + padding - size)/stride + 1;
        this.outH = (height + padding - size)/stride + 1;
        this.outC = c;
        this.outputs = outH * outW * outC;
        this.inputs = height*width*c;
        this.size = size;
        this.stride = stride;

        final int outputSize = outH * outW * outC * batch;
        this.indexes = new IntArray(outputSize);
        this.output = new FloatArray(outputSize);
        this.delta =  new FloatArray(outputSize);

        System.out.printf("Max         %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, outW, outH, outC);
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

        int wOffset = -pad/2;
        int hOffset = -pad/2;

        for(int b = 0; b < batch; ++b){

            final int cBbase = b * this.c;
            IntStream.range(0,this.c).parallel().forEach(k -> {

                final int indexB = this.h*(k + cBbase);
                final int cB = outH*(k + cBbase);

                for(int i = 0; i < outH; ++i){

                    final int outIndexBase = outW*(i + cB);
                    final int curHbase = hOffset + i*stride;

                    for(int j = 0; j < outW; ++j){
                        final int outIndex = outIndexBase + j;
                        final int curWbase = wOffset + j*stride;

                        float max = -Rand.MAX_FLOAT;
                        int maxI = -1;

                        for(int n = 0; n < size; ++n){

                            final int curH = curHbase + n;
                            final int indexBase = this.w*(curH + indexB);
                            final boolean isCurHOk = curH >= 0 && curH < this.h;

                            for(int m = 0; m < size; ++m){

                                final int curW = curWbase + m;
                                final int index = curW + indexBase;

                                final float val = (isCurHOk && curW >= 0 && curW < this.w) ? net.input.get(index) : -Rand.MAX_FLOAT;

                                if(val > max) {
                                    maxI = index;
                                    max = val;
                                }
                            }
                        }
                        output.set(outIndex,max);
                        indexes.set(outIndex,maxI);
                    }
                }
            });

//            for(int k = 0; k < c; ++k){
//                for(int i = 0; i < h; ++i){
//                    for(int j = 0; j < w; ++j){
//                        int outIndex = j + w*(i + h*(k + c*b));
//                        float max = -Rand.MAX_FLOAT;
//                        int maxI = -1;
//                        for(int n = 0; n < size; ++n){
//                            for(int m = 0; m < size; ++m){
//                                int curH = hOffset + i*stride + n;
//                                int curW = wOffset + j*stride + m;
//                                int index = curW + this.w*(curH + this.h*(k + b*this.c));
//                                boolean valid = (curH >= 0 && curH < this.h && curW >= 0 && curW < this.w);
//
//                                float val = (valid) ? net.input.get(index) : -Rand.MAX_FLOAT;
//
//                                maxI = (val > max) ? index : maxI;
//                                max   = (val > max) ? val   : max;
//                            }
//                        }
//                        output.set(outIndex,max);
//                        indexes.set(outIndex,maxI);
//                    }
//                }
//            }
        }
    }

    public void backward(Network net) {

        int i;
        int h = outH;
        int w = outW;
        int c = this.c;

        for(i = 0; i < h*w*c*batch; ++i){

            int index = indexes.get(i);
            net.delta.set(index,net.delta.get(index) + delta.get(i));
        }
    }
}
