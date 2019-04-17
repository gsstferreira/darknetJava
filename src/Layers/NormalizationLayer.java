package Layers;

import Classes.Buffers.FloatBuffer;
import Classes.Layer;
import Classes.Network;
import Enums.LayerType;
import Tools.Blas;
import Tools.Buffers;


public class NormalizationLayer extends Layer {

    public NormalizationLayer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa) {

        System.out.printf("Local Response Normalization Layer: %d x %d x %d image, %d size\n", w,h,c,size);

        this.type = LayerType.NORMALIZATION;
        this.batch = batch;
        this.h = this.outH = h;
        this.w = this.outW = w;
        this.c = this.outC = c;
        this.kappa = kappa;
        this.size = size;
        this.alpha = alpha;
        this.beta = beta;
        this.output = new FloatBuffer(h * w * c * batch);
        this.delta = new FloatBuffer(h * w * c * batch);
        this.squared = new FloatBuffer(h * w * c * batch);
        this.norms = new FloatBuffer(h * w * c * batch);
        this.inputs = w*h*c;
        this.outputs = this.inputs;
    }

    public void resize(int w, int h) {
        
        int c = this.c;
        int batch = this.batch;
        this.h = h;
        this.w = w;
        this.outH = h;
        this.outW = w;
        this.inputs = w*h*c;
        this.outputs = this.inputs;
        
        this.output = Buffers.realloc(this.output, h * w * c * batch);
        this.delta = Buffers.realloc(this.delta, h * w * c * batch);
        this.squared = Buffers.realloc(this.squared, h * w * c * batch);
        this.norms = Buffers.realloc(this.norms, h * w * c * batch);
    }

    public void forward(Network net) {
        
        int k,b;
        int w = this.w;
        int h = this.h;
        int c = this.c;
        
        Blas.scalCpu(w*h*c*this.batch, 0, this.squared, 1);

        for(b = 0; b < this.batch; ++b){
            FloatBuffer squared = this.squared.offsetNew(w*h*c*b);
            FloatBuffer norms   = this.norms.offsetNew(w*h*c*b);;
            FloatBuffer input   = net.input.offsetNew(w*h*c*b);

            Blas.powCpu(w*h*c, 2, input, 1, squared, 1);
            Blas.constCpu(w*h, this.kappa, norms, 1);

            for(k = 0; k < this.size/2; ++k){

                FloatBuffer squared2 = squared.offsetNew(w*h*k);
                Blas.axpyCpu(w*h, this.alpha, squared2, 1, norms, 1);
            }

            for(k = 1; k < this.c; ++k){

                FloatBuffer norms2 = norms.offsetNew(w*h*k);
                FloatBuffer norms3 = norms.offsetNew(w*h*(k-1));

                Blas.copyCpu(w*h, norms3, 1, norms2, 1);
                int prev = k - ((this.size-1)/2) - 1;
                int next = k + (this.size/2);

                if(prev >= 0) {
                    FloatBuffer squared2 = squared.offsetNew(w*h*prev);
                    Blas.axpyCpu(w*h, -this.alpha, squared2, 1, norms2, 1);
                }

                if(next < this.c) {
                    FloatBuffer squared2 = squared.offsetNew(w*h*next);
                    Blas.axpyCpu(w*h,  this.alpha, squared2, 1, norms2, 1);
                }
            }
        }
        Blas.powCpu(w*h*c*this.batch, -this.beta, this.norms, 1, this.output, 1);
        Blas.mulCpu(w*h*c*this.batch, net.input, 1, this.output, 1);
    }

    public void backward(Network net) {
        
        int w = this.w;
        int h = this.h;
        int c = this.c;

        Blas.powCpu(w*h*c*this.batch, -this.beta, this.norms, 1, net.delta, 1);
        Blas.mulCpu(w*h*c*this.batch, this.delta, 1, net.delta, 1);
    }
}
