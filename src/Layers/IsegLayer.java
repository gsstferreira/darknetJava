package Layers;

import Classes.Layer;
import Classes.Network;
import Enums.LayerType;
import Tools.Blas;
import Tools.BufferUtil;
import Tools.Util;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

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
        this.cost = BufferUtils.createFloatBuffer(1);
        this.outputs = h*w*this.c;
        this.inputs = this.outputs;
        this.truths = 90*(this.w*this.h+1);
        this.delta = BufferUtils.createFloatBuffer(batch*this.outputs);
        this.output = BufferUtils.createFloatBuffer(batch*this.outputs);

        this.counts = BufferUtils.createIntBuffer(90);
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

        output = BufferUtil.reallocBuffer(output, batch*outputs);
        delta = BufferUtil.reallocBuffer(delta, batch*outputs);
    }

    public void forward(Network net) {

        long time = Util.getTime();

        int i,b,j,k;
        int ids = this.extra;

        BufferUtil.bufferCopy(net.input,output,outputs*batch);
        BufferUtil.setBufferValue(delta,0,outputs*batch);

        for (b = 0; b < this.batch; ++b){
            // a priori, each pixel has no class
            for(i = 0; i < this.classes; ++i){
                for(k = 0; k < this.w*this.h; ++k){
                    int index = b*this.outputs + i*this.w*this.h + k;

                    delta.put(index, -output.get(index));
                }
            }

            // a priori, embedding should be small magnitude
            for(i = 0; i < ids; ++i){
                for(k = 0; k < this.w*this.h; ++k){

                    int index = b*this.outputs + (i+this.classes)*this.w*this.h + k;
                    delta.put(index, -0.1f*output.get(index));
                }
            }
            BufferUtil.setBufferValue(counts,0,90);

            for(i = 0; i < 90; ++i){

                Blas.fillCpu(ids, 0, FloatBuffer.wrap(this.sums[i]), 1);
                int c = (int) net.truth.get(b*this.truths + i*(this.w*this.h+1));

                if(c < 0) {
                    break;
                }

                // add up metric embeddings for each instance
                for(k = 0; k < this.w*this.h; ++k){
                    int index = b*this.outputs + c*this.w*this.h + k;
                    float v = net.truth.get(b*this.truths + i*(this.w*this.h + 1) + 1 + k);
                    if(v != 0) {

                        delta.put(index,v - output.get(index));
                        FloatBuffer fb = BufferUtil.offsetBuffer(output,b*this.outputs + this.classes*this.w*this.h + k);

                        Blas.axpyCpu(ids, 1, fb, this.w*this.h, FloatBuffer.wrap(this.sums[i]), 1);
                        counts.put(i,counts.get(i) + 1);
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
                Blas.scalCpu(ids, 1.f/this.counts.get(i), FloatBuffer.wrap(this.sums[i]), 1);
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
                                    delta.put(index,val);
                                }
                                else {

                                    float val = delta.get(index) - ((diff < 0) ? -0.1f : 0.1f);
                                    delta.put(index,val);
                                }
                            }
                        }
                    }
                }
            }

            for(i = 0; i < ids; ++i){
                for(k = 0; k < this.w*this.h; ++k){
                    int index = b*this.outputs + (i+this.classes)*this.w*this.h + k;

                    delta.put(index,delta.get(index) * 0.1f);
                }
            }
        }
        this.cost.put(0,(float)Math.pow(Util.magArray(this.delta, this.outputs * this.batch), 2));

        long time2 = Util.getTime();
        System.out.print(String.format("took %f sec\n",(time2 - time)/1000.0f));
    }

    public void backward(Network net) {

        Blas.axpyCpu(this.batch*this.inputs, 1, this.delta, 1, net.delta, 1);
    }
}
