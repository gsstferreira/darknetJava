package Yolo.Layers;

import Classes.Arrays.FloatArray;
import Classes.Arrays.IntArray;
import Classes.Layer;
import Classes.Network;
import Tools.Blas;
import Tools.Buffers;
import Tools.Util;
import Yolo.Enums.LayerType;



public class IsegLayer extends Layer {

    public IsegLayer(int batch, int w, int h, int classes, int ids) {
        
        this.type = LayerType.ISEG;

        this.h = h;
        this.w = w;
        this.c = classes + ids;
        this.outW = this.w;
        this.outH = this.h;
        this.outC = this.c;
        this.classes = classes;
        this.batch = batch;
        this.extra = ids;
        this.cost = new FloatArray(1);
        this.outputs = h*w*this.c;
        this.inputs = this.outputs;
        this.truths = 90*(this.w*this.h+1);
        this.delta = new FloatArray(batch*this.outputs);
        this.output = new FloatArray(batch*this.outputs);

        this.counts = new IntArray(90);
        this.sums = new float[90][];
        if(ids != 0){
            int i;
            for(i = 0; i < 90; ++i){
                this.sums[i] = new float[ids];
            }
        }
    }

    public void resize(int width, int height) {
        
        w = width;
        h = height;

        outputs = height*width*c;
        inputs = outputs;

        output = Buffers.realloc(output, batch*outputs);
        delta = Buffers.realloc(delta, batch*outputs);
    }

    public void forward(Network net) {

        long time = System.currentTimeMillis();

        int i,b,j,k;
        int ids = this.extra;

        Buffers.copy(net.input,output,outputs*batch);
        delta.setAll(0,outputs*batch);

        for (b = 0; b < this.batch; ++b){
            // a priori, each pixel has no class
            for(i = 0; i < this.classes; ++i){
                for(k = 0; k < this.w*this.h; ++k){
                    int index = b*this.outputs + i*this.w*this.h + k;

                    delta.set(index, -output.get(index));
                }
            }

            // a priori, embedding should be small magnitude
            for(i = 0; i < ids; ++i){
                for(k = 0; k < this.w*this.h; ++k){

                    int index = b*this.outputs + (i+this.classes)*this.w*this.h + k;
                    delta.set(index, -0.1f*output.get(index));
                }
            }
            counts.setAll(0,90);

            for(i = 0; i < 90; ++i){

                Blas.fillCpu(ids, 0, new FloatArray(this.sums[i]), 1);
                int c = (int) net.truth.get(b*this.truths + i*(this.w*this.h+1));

                if(c < 0) {
                    break;
                }

                // add up metric embeddings for each instance
                for(k = 0; k < this.w*this.h; ++k){
                    int index = b*this.outputs + c*this.w*this.h + k;
                    float v = net.truth.get(b*this.truths + i*(this.w*this.h + 1) + 1 + k);
                    if(v != 0) {

                        delta.set(index,v - output.get(index));
                        FloatArray fb = output.offsetNew(b*this.outputs + this.classes*this.w*this.h + k);

                        Blas.axpyCpu(ids, 1, fb, this.w*this.h, new FloatArray(this.sums[i]), 1);
                        counts.set(i,counts.get(i) + 1);
                    }
                }
            }

            float[] mse = new float[90];

            for(i = 0; i < 90; ++i){
                int c = (int) net.truth.get(b*this.truths + i*(this.w*this.h+1));

                if(c < 0) {
                    break;
                }
                for(k = 0; k < this.w*this.h; ++k){
                    float v = net.truth.get(b*this.truths + i*(this.w*this.h + 1) + 1 + k);
                    if(v != 0) {

                        int z;
                        float sum = 0;
                        for(z = 0; z < ids; ++z){
                            int index = b*this.outputs + (this.classes + z)*this.w*this.h + k;
                            sum += Math.pow(this.sums[i][z]/this.counts.get(i) - this.output.get(index), 2);
                        }
                        mse[i] += sum;
                    }
                }
                mse[i] /= this.counts.get(i);
            }

            // Calculate average embedding
            for(i = 0; i < 90; ++i){
                if(this.counts.get(i) == 0) {
                    continue;
                }
                Blas.scalCpu(ids, 1.f/this.counts.get(i), new FloatArray(this.sums[i]), 1);
                if(b == 0 && net.gpuIndex == 0) {

                    System.out.print(String.format("%4d, %6.3f, ", this.counts.get(i), mse[i]));
                    for(j = 0; j < ids; ++j){
                        System.out.print(String.format("%6.3f,", this.sums[i][j]));
                    }
                    System.out.println();
                }
            }

            for(i = 0; i < 90; ++i){
                if(this.counts.get(i) == 0) {
                    continue;
                }
                for(k = 0; k < this.w*this.h; ++k){
                    float v = net.truth.get(b*this.truths + i*(this.w*this.h + 1) + 1 + k);
                    if(v != 0){
                        for(j = 0; j < 90; ++j){
                            if(this.counts.get(j) == 0) {
                                continue;
                            }
                            int z;
                            for(z = 0; z < ids; ++z){
                                int index = b*this.outputs + (this.classes + z)*this.w*this.h + k;
                                float diff = this.sums[j][z] - this.output.get(index);
                                if (j == i) {

                                    float val = delta.get(index) + ((diff < 0) ? -0.1f : 0.1f);
                                    delta.set(index,val);
                                }
                                else {

                                    float val = delta.get(index) - ((diff < 0) ? -0.1f : 0.1f);
                                    delta.set(index,val);
                                }
                            }
                        }
                    }
                }
            }

            for(i = 0; i < ids; ++i){
                for(k = 0; k < this.w*this.h; ++k){
                    int index = b*this.outputs + (i+this.classes)*this.w*this.h + k;

                    delta.set(index,delta.get(index) * 0.1f);
                }
            }
        }
        this.cost.set(0,(float)Math.pow(Util.magArray(this.delta, this.outputs * this.batch), 2));

        long time2 = System.currentTimeMillis();
        System.out.print(String.format("took %f sec\n",(time2 - time)/1000.0f));
    }

    public void backward(Network net) {

        Blas.axpyCpu(this.batch*this.inputs, 1, this.delta, 1, net.delta, 1);
    }
}
