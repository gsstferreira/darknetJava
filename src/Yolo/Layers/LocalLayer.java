package Yolo.Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Classes.UpdateArgs;
import Tools.Blas;
import Tools.Gemm;
import Tools.ImCol;
import Tools.Rand;
import Yolo.Enums.Activation;
import Yolo.Enums.LayerType;


public class LocalLayer extends Layer {

    private int localOutHeight() {
        
        int h = this.h;
        if (this.pad == 0) {
            h -= this.size;
        }
        else h -= 1;
        return h/this.stride + 1;
    }

    private int localOutWidth() {
        
        int w = this.w;
        if (this.pad == 0) {
            w -= this.size;
        }
        else w -= 1;
        return w/this.stride + 1;
    }

    public LocalLayer(int batch, int h, int w, int c, int n, int size, int stride, int pad, Activation activation) {

        int i;
        this.type = LayerType.LOCAL;

        this.h = h;
        this.w = w;
        this.c = c;
        this.n = n;
        this.batch = batch;
        this.stride = stride;
        this.size = size;
        this.pad = pad;

        int outH = localOutHeight();
        int outW = localOutWidth();
        int locations = outH*outW;
        this.outH = outH;
        this.outW = outW;
        this.outC = n;
        this.outputs = this.outH * this.outW * this.outC;
        this.inputs = this.w * this.h * this.c;

        this.weights = new FloatBuffer(c*n*size*size*locations);
        this.weightUpdates = new FloatBuffer(c*n*size*size*locations);

        this.biases = new FloatBuffer(this.outputs);
        this.biasUpdates = new FloatBuffer(this.outputs);

        float scale = (float) Math.sqrt(2./(size*size*c));
        for(i = 0; i < c*n*size*size; ++i) {
            this.weights.put(i,scale* Rand.randUniform(-1, 1));
        }

        this.output = new FloatBuffer(this.batch*outH*outW*n);
        this.delta  = new FloatBuffer(this.batch*outH*outW*n);

        this.workspaceSize = outH*outW*size*size*c;
        this.activation = activation;
    }

    public void forward(Network net) {

        int out_h = localOutHeight();
        int out_w = localOutWidth();
        int locations = out_h * out_w;

        for(int i = 0; i < batch; ++i){
            
            FloatBuffer fb = output.offsetNew(i*outputs);
            
            Blas.copyCpu(outputs, biases, 1, fb, 1);
        }

        for(int i = 0; i < batch; ++i){

            FloatBuffer input = net.input.offsetNew(i*w*h*c);
            ImCol.im2ColCpu(input, c, h, w, size, stride, pad, net.workspace);

            FloatBuffer _output = output.offsetNew(i*outputs);

            for(int j = 0; j < locations; ++j){


                FloatBuffer _a = weights.offsetNew(j*size*size*c*n);
                FloatBuffer _b = net.workspace.offsetNew(j);
                FloatBuffer _c = _output.offsetNew(j);

                int m = n;
                int n = 1;
                int k = size*size*c;

                Gemm.gemm(0,0,m,n,k,1,_a,k,_b,locations,1,_c,locations);
            }
        }
        Activation.activateArray(output, outputs*batch, activation);
    }

    public void backward(Network net) {

        int i, j;
        int locations = outW*outH;

        Activation.gradientArray(output, outputs*batch, activation, delta);

        for(i = 0; i < batch; ++i){

            FloatBuffer fb = delta.offsetNew(i*outputs);
            Blas.axpyCpu(outputs, 1, fb, 1, biasUpdates, 1);
        }

        for(i = 0; i < batch; ++i){

            FloatBuffer input = net.input.offsetNew(i*w*h*c);

            ImCol.im2ColCpu(input, c, h, w, size, stride, pad, net.workspace);

            for(j = 0; j < locations; ++j){
                FloatBuffer _a = delta.offsetNew(i*outputs + j);
                FloatBuffer _b = net.workspace.offsetNew(j);
                FloatBuffer _c = weightUpdates.offsetNew(j*size*size*c*n);

                int m = n;
                int n = size*size*c;
                int k = 1;

                Gemm.gemm(0,1,m,n,k,1,_a,locations,_b,locations,1,_c,n);
            }

            if(net.delta != null){
                for(j = 0; j < locations; ++j){

                    FloatBuffer _a = weights.offsetNew(j*size*size*c*n);
                    FloatBuffer _b = delta.offsetNew(i*outputs + j);
                    FloatBuffer _c = net.workspace.offsetNew(j);

                    int m = size*size*c;
                    int n = 1;

                    Gemm.gemm(1,0,m,n, n,1,_a,m,_b,locations,0,_c,locations);
                }

                FloatBuffer fb = net.delta.offsetNew(i*c*h*w);
                ImCol.col2ImCpu(net.workspace, c, h,  w,  size,  stride, pad, fb);
            }
        }
    }

    public void update(UpdateArgs a) {

        float learning_rate = a.learningRate * learningRateScale;
        float momentum = a.momentum;
        float decay = a.decay;
        int batch = a.batch;

        int locations = outW*outH;
        int _size = size*size*c*n*locations;

        Blas.axpyCpu(outputs,learning_rate/batch,biasUpdates,1,biases,1);
        Blas.scalCpu(outputs,momentum,biasUpdates,1);

        Blas.axpyCpu(_size, -decay*batch,weights,1,weightUpdates,1);
        Blas.axpyCpu(_size,learning_rate/batch,weightUpdates,1,weights,1);
        Blas.scalCpu(_size,momentum,weightUpdates,1);
    }
}
