package Tools;

import Classes.Arrays.FloatArray;
import java.util.stream.IntStream;

public abstract class Blas {

    private static final int parallelLength = 7500;

    public static void reorgCpu(FloatArray x, int w, int h, int c, int batch, int stride, int forward, FloatArray out) {

        int out_c = c/(stride*stride);

        for(int b = 0; b < batch; ++b){
            for(int k = 0; k < c; ++k){
                for(int j = 0; j < h; ++j){
                    for(int i = 0; i < w; ++i){

                        int in_index  = i + w*(j + h*(k + c*b));
                        int c2 = k % out_c;
                        int offset = k / out_c;
                        int w2 = i*stride + offset % stride;
                        int h2 = j*stride + offset / stride;
                        int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

                        if(forward != 0) {
                            out.set(out_index,x.get(in_index));
                        }
                        else  {
                            out.set(in_index,x.get(out_index));
                        }
                    }
                }
            }
        }
    }

    public static void flatten(FloatArray x, int size, int layers, int batch, int forward) {

        FloatArray swap = new FloatArray(size*layers*batch);

        for(int b = 0; b < batch; ++b){
            for(int c = 0; c < layers; ++c){
                for(int i = 0; i < size; ++i){

                    int i1 = b*layers*size + c*size + i;
                    int i2 = b*layers*size + i*layers + c;

                    if (forward != 0) {
                        swap.set(i2,x.get(i1));
                    }
                    else  {
                        swap.set(i1,x.get(i2));
                    }
                }
            }
        }
        Buffers.copy(swap,x,size*layers*batch);
    }

    public static void weightedSumCpu(FloatArray a, FloatArray b, FloatArray s, int n, FloatArray c) {

        for(int i = 0; i < n; ++i){

            float val = s.get(i)*a.get(i) + (1 - s.get(i))*(b != null ? b.get(i) : 0);
            c.set(i,val);
        }
    }

//    public static void weightedDeltaCpu(FloatArray a, FloatArray b, FloatArray s, FloatArray da, FloatArray db, FloatArray ds, int n, FloatArray dc) {
//
//        for(int i = 0; i < n; ++i){
//
//            if(da != null) {
//                da.set(i,da.get(i) + dc.get(i)*s.get(i));
//            }
//
//            if(db != null) {
//                db.set(i,db.get(i) + dc.get(i)*(1 - s.get(i)));
//            }
//            ds.set(i,ds.get(i) + dc.get(i)*(a.get(i) - b.get(i)));
//        }
//    }

    public static void shortcutCpu(int batch, int w1, int h1, int c1, FloatArray add, int w2, int h2, int c2, float s1, float s2, FloatArray out) {

        int stride = w1/w2;
        int sample = w2/w1;

        assert(stride == h1/h2);
        assert(sample == h2/h1);

        if(stride < 1) stride = 1;
        if(sample < 1) sample = 1;

        int minw = (w1 < w2) ? w1 : w2;
        int minh = (h1 < h2) ? h1 : h2;
        int minc = (c1 < c2) ? c1 : c2;

        for(int b = 0; b < batch; ++b){
            for(int k = 0; k < minc; ++k){
                for(int j = 0; j < minh; ++j){
                    for(int i = 0; i < minw; ++i){
                        int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                        int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));

                        float val = s1 * out.get(out_index) + s2*add.get(add_index);
                        out.set(out_index,val);
                    }
                }
            }
        }
    }

    public static void meanCpu(FloatArray x, int batch, int filters, int spatial, FloatArray mean) {

        float scale = 1.0f/(batch * spatial);

        for(int i = 0; i < filters; ++i){

            mean.set(i,0);
            for(int j = 0; j < batch; ++j){
                for(int k = 0; k < spatial; ++k){
                    int index = j*filters*spatial + i*spatial + k;

                    mean.set(i,mean.get(i) + x.get(index));
                }
            }
            mean.set(i,mean.get(i)* scale);
        }
    }

    public static void varianceCpu(FloatArray x, FloatArray mean, int batch, int filters, int spatial, FloatArray variance) {

        float scale = 1.0f/(batch * spatial - 1);

        for(int i = 0; i < filters; ++i){

            variance.set(i,0);

            for(int j = 0; j < batch; ++j){
                for(int k = 0; k < spatial; ++k){
                    int index = j*filters*spatial + i*spatial + k;

                    float val = variance.get(i) + (float) Math.pow((x.get(index) - mean.get(i)),2);

                    variance.set(i,val);
                }
            }
            variance.set(i,variance.get(i)*scale);
        }
    }

    public static void l2normalizeCpu(FloatArray x, FloatArray dx, int batch, int filters, int spatial) {

        for(int b = 0; b < batch; ++b){
            for(int i = 0; i < spatial; ++i){
                float sum = 0;
                for(int f = 0; f < filters; ++f){
                    int index = b*filters*spatial + f*spatial + i;
                    sum += (float) Math.pow(x.get(index), 2);
                }

                sum = (float) Math.sqrt(sum);

                for(int f = 0; f < filters; ++f){
                    int index = b*filters*spatial + f*spatial + i;

                    x.set(index,x.get(index)/sum);
                    dx.set(index,(1 - x.get(index))/sum);
                }
            }
        }
    }

    public static void normalizeCpu(FloatArray x, FloatArray mean, FloatArray variance, int batch, int filters, int spatial) {

        for(int b = 0; b < batch; ++b){

            final int bFilters = b * filters *spatial;
            IntStream.range(0,filters).parallel().forEach(f -> {

                final int indexBase = bFilters + f*spatial;
                final float sqrtVar = (float) (Math.sqrt(variance.get(f)) + 0.000001);
                final float meanF = mean.get(f);

                for(int i = 0; i < spatial; ++i){

                    final int index = indexBase + i;
                    float val = (x.get(index) - meanF)/sqrtVar;

                    x.set(index,val);
                }
            });

//            for(int f = 0; f < filters; ++f){
//                for(int i = 0; i < spatial; ++i){
//                    int index = b*filters*spatial + f*spatial + i;
//
//                    double  val = (x.get(index) - mean.get(f))/(Math.sqrt(variance.get(f)) + 0.000001);
//                    x.set(index,(float)val);
//                }
//            }
        }
    }

    public static void constCpu(int N, float ALPHA, FloatArray X, int INCX) {

        for(int i = 0; i < N; ++i) {
            X.set(i*INCX,ALPHA);
        }
    }

    public static void mulCpu(int N, FloatArray X, int INCX, FloatArray Y, int INCY) {

        for(int i = 0; i < N; ++i) {

            Y.set(i*INCY, Y.get(i*INCY) * X.get(i*INCX));
        }
    }

    public static void powCpu(int N, float ALPHA, FloatArray X, int INCX, FloatArray Y, int INCY) {

        for(int i = 0; i < N; ++i) {

            Y.set(i*INCY,(float) Math.pow(X.get(i*INCX),ALPHA));
        }
    }

    public static void axpyCpu(int N, float ALPHA, FloatArray X, int INCX, FloatArray Y, int INCY) {

        if(N > parallelLength) {
            IntStream.range(0,N).parallel().forEach(i -> Y.addIn(i*INCY,ALPHA * X.get(i*INCX)));
        }
        else {
            for(int i = 0; i < N; i++) {
                Y.addIn(i*INCY,ALPHA * X.get(i*INCX));
            }
        }
    }

    public static void scalCpu(int N, float ALPHA, FloatArray X, int INCX) {

        if(N > parallelLength) {
            IntStream.range(0,N).parallel().forEach(i -> X.mulIn(i*INCX,ALPHA));
        }
        else {
            for(int i = 0; i < N; i++) {
                X.mulIn(i*INCX,ALPHA);
            }
        }
    }

    public static void fillCpu(int N, float ALPHA, FloatArray X, int INCX) {

        IntStream.range(0,N).parallel().forEach(i -> X.set(i*INCX,ALPHA));
    }

//    public static void deinterCpu(int NX, FloatArray X, int NY, FloatArray Y, int B, FloatArray OUT) {
//
//        int index = 0;
//
//        for(int j = 0; j < B; ++j) {
//            for(int i = 0; i < NX; ++i){
//
//                if(X != null) {
//                    X.set(j*NX + i,X.get(j*NX + i) + OUT.get(index));
//                }
//                ++index;
//            }
//
//            for(int i = 0; i < NY; ++i){
//
//                if(Y != null) {
//                    Y.set(j*NY +i, Y.get(j*NY + i) + OUT.get(index));
//                }
//                ++index;
//            }
//        }
//    }
//
//    public static void interCpu(int NX, FloatArray X, int NY, FloatArray Y, int B, FloatArray OUT) {
//
//        int index = 0;
//
//        for(int j = 0; j < B; ++j) {
//            for(int i = 0; i < NX; ++i){
//                OUT.set(index,X.get(j*NX + i));
//                index++;
//            }
//
//            for(int i = 0; i < NY; ++i){
//                OUT.set(index,Y.get(j*NY + i));
//                index++;
//            }
//        }
//    }



//    public static void multAddIntoCpu(int N, FloatArray X, FloatArray Y, FloatArray Z) {
//
//        for(int i = 0; i < N; ++i) {
//            float val = Z.get(i) + X.get(i) * Y.get(i);
//            Z.set(i,val);
//        }
//    }

    public static void smoothL1Cpu(int n, FloatArray pred, FloatArray truth, FloatArray delta, FloatArray error) {

        for(int i = 0; i < n; ++i){

            float diff = truth.get(i) - pred.get(i);
            float abs_val = Math.abs(diff);
            if(abs_val < 1) {
                error.set(i,diff*diff);
                delta.set(i,diff);
            }
            else {
                error.set(i,2 * abs_val - 1);
                delta.set(i,(diff < 0) ? 1 : -1);
            }
        }
    }

    public static void l1Cpu(int n, FloatArray pred, FloatArray truth, FloatArray delta, FloatArray error) {

        for(int i = 0; i < n; ++i){
            float diff = truth.get(i) - pred.get(i);
            error.set(i,Math.abs(diff));
            delta.set(i,(diff > 0) ? 1 : -1);
        }
    }

    public static void softmaxXEntCpu(int n, FloatArray pred, FloatArray truth, FloatArray delta, FloatArray error) {

        for(int i = 0; i < n; ++i){
            float t = truth.get(i);
            float p = pred.get(i);

            error.set(i,(t != 0) ? - (float)Math.log(p) : 0);
            delta.set(i,t - p);
        }
    }

    public static void logisticXEntCpu(int n, FloatArray pred, FloatArray truth, FloatArray delta, FloatArray error) {

        for(int i = 0; i < n; ++i){
            float t = truth.get(i);
            float p = pred.get(i);

            double val = -t*Math.log(p) - (1-t)*Math.log(1-p);

            error.set(i,(float) val);
            delta.set(i,t - p);
        }
    }

    public static void l2Cpu(int n, FloatArray pred, FloatArray truth, FloatArray delta, FloatArray error) {

        for(int i = 0; i < n; ++i){
            float diff = truth.get(i) - pred.get(i);
            error.set(i,diff * diff);
            delta.set(i,diff);
        }
    }

//    public static float dotCpu(int N, FloatArray X, int INCX, FloatArray Y, int INCY) {
//
//        float dot = 0;
//
//        for(int i = 0; i < N; ++i) {
//
//            dot += X.get(i*INCX) * Y.get(i*INCY);
//        }
//        return dot;
//    }

    public static void softmax(FloatArray input, int n, float temp, int stride, FloatArray output) {

        float sum = 0;
        float largest = - Rand.MAX_FLOAT;

        for(int i = 0; i < n; ++i){
            if(input.get(i*stride) > largest)  {
                largest = input.get(i*stride);
            }
        }

        for(int i = 0; i < n; ++i){
            float e = (float) Math.exp(input.get(i*stride) / temp - largest / temp);
            sum += e;
            output.set(i*stride,e);
        }

        for(int i = 0; i < n; ++i){
            output.set(i*stride,output.get(i*stride)/sum);
        }
    }

    public static void softmaxCpu(FloatArray input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, FloatArray output) {

        for(int b = 0; b < batch; ++b){
            for(int g = 0; g < groups; ++g){

                FloatArray fb1 = input.offsetNew(b*batch_offset + g*group_offset);
                FloatArray fb2 = output.offsetNew(b*batch_offset + g*group_offset);

                softmax(fb1, n, temp, stride, fb2);
            }
        }
    }

    public static void upsampleCpu(FloatArray in, int w, int h, int c, int batch, int stride, int forward, float scale, FloatArray out) {

        final int mWh = w*h;
        final int mWhc = mWh * c;
        final int wStride = w*stride;
        final int mStride = mWhc * stride * stride;
        final int mStride2 = mWh * stride * stride;

        for(int b = 0; b < batch; ++b){

            final int bWhc = b * mWhc;
            final int bStride = b * mStride;

            IntStream.range(0,c).parallel().forEach(k ->{

                final int inB = bWhc + k * mWh;
                final int outB = bStride + k * mStride2;

                for(int j = 0; j < h*stride; ++j){

                    final int inBase = inB + (j/stride)*w;
                    final int outBase = outB + j*wStride;

                    if(forward != 0) {
                        for(int i = 0; i < wStride; ++i){

                            out.set(outBase + i,scale*in.get(inBase + i/stride));
                        }
                    }
                    else {
                        for(int i = 0; i < wStride; ++i){

                            in.addIn(inBase + i/stride,scale*out.get(outBase + i));
                        }
                    }
                }
            });

//            for(int k = 0; k < c; ++k){
//                for(int j = 0; j < h*stride; ++j){
//                    for(int i = 0; i < w*stride; ++i){
//                        int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
//                        int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
//
//                        if(forward != 0) {
//                            out.set(out_index,scale*in.get(in_index));
//                        }
//                        else {
//                            in.set(in_index, in.get(in_index) + scale*out.get(out_index));
//                        }
//                    }
//                }
//            }
        }
    }
}
