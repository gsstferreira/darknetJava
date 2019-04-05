package Layers;

import Classes.Layer;
import Classes.Network;
import Enums.LayerType;
import Tools.Blas;
import Tools.Buffers;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

public class NormalizationLayer extends Layer {

    public NormalizationLayer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa) {

        this.type = LayerType.NORMALIZATION;
        this.batch = batch;
        this.h = this.outH = h;
        this.w = this.outW = w;
        this.c = this.outC = c;
        this.kappa = kappa;
        this.size = size;
        this.alpha = alpha;
        this.beta = beta;
        this.output = Buffers.newBufferF(h * w * c * batch);
        this.delta = Buffers.newBufferF(h * w * c * batch);
        this.squared = Buffers.newBufferF(h * w * c * batch);
        this.norms = Buffers.newBufferF(h * w * c * batch);
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
            FloatBuffer squared = Buffers.offset(this.squared,w*h*c*b);
            FloatBuffer norms   = Buffers.offset(this.norms,w*h*c*b);;
            FloatBuffer input   = Buffers.offset(net.input,w*h*c*b);

            Blas.powCpu(w*h*c, 2, input, 1, squared, 1);
            Blas.constCpu(w*h, this.kappa, norms, 1);

            for(k = 0; k < this.size/2; ++k){

                FloatBuffer squared2 = Buffers.offset(squared,w*h*k);
                Blas.axpyCpu(w*h, this.alpha, squared2, 1, norms, 1);
            }

            for(k = 1; k < this.c; ++k){

                FloatBuffer norms2 = Buffers.offset(norms,w*h*k);
                FloatBuffer norms3 = Buffers.offset(norms,w*h*(k-1));

                Blas.copyCpu(w*h, norms3, 1, norms2, 1);
                int prev = k - ((this.size-1)/2) - 1;
                int next = k + (this.size/2);

                if(prev >= 0) {
                    FloatBuffer squared2 = Buffers.offset(squared,w*h*prev);
                    Blas.axpyCpu(w*h, -this.alpha, squared2, 1, norms2, 1);
                }

                if(next < this.c) {
                    FloatBuffer squared2 = Buffers.offset(squared,w*h*next);
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
