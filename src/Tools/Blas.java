package Tools;


// Completo
public abstract class Blas {

    public static void reorgCpu(float[] x, int w, int h, int c, int batch, int stride, int forward, float[] out, int offX, int offO) {

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
                            out[offO + out_index] = x[offX + in_index];
                        }
                        else  {
                            out[offO + in_index] = x[offX + out_index];
                        }
                    }
                }
            }
        }
    } //

    public static void flatten(float[] x, int size, int layers, int batch, int forward, int offX) {

        float[] swap = new float[size*layers*batch];

        for(int b = 0; b < batch; ++b){
            for(int c = 0; c < layers; ++c){
                for(int i = 0; i < size; ++i){

                    int i1 = b*layers*size + c*size + i;
                    int i2 = b*layers*size + i*layers + c;

                    if (forward != 0) {
                        swap[i2] = x[offX + i1];
                    }
                    else  {
                        swap[i1] = x[offX + i2];
                    }
                }
            }
        }
        System.arraycopy(swap,0,x,offX,swap.length);
    } //

    public static void weightedSumCpu(float[] a, float[] b, float[] s, int n, float[] c, int offA, int offB, int offS, int offC) {

        for(int i = 0; i < n; ++i){

            c[offC + i] = s[offS + i] * a[offA + i];

            if(b != null) {
                c[offC + i] += (1 - s[offS + i]) * b[offB + i];
            }
        }
    } //

    public static void weightedDeltaCpu(float[] a, float[] b, float[] s, float[] da, float[] db, float[] ds, int n,
                                        float[] dc, int oA, int oB, int oS, int oDa, int oDb, int oDs, int oDc) {

        for(int i = 0; i < n; ++i){

            if(da != null) {
                da[oDa + i] += dc[oDc + i] * s[oS + i];
            }

            if(db != null) {
                db[oDb + i] += dc[oDc + i] * (1 - s[oS + i]);
            }

            ds[oDs + i] += dc[oDc + i] * (a[oA + i] - b[oB + i]);
        }
    } //

    public static void shortcutCpu(int batch, int w1, int h1, int c1, float[] add, int w2, int h2, int c2, float s1, float s2, float[] out, int oAdd, int oOut) {

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
                        out[oOut + out_index] = s1 * out[oOut + out_index] + s2 * add[oAdd+ + add_index];
                    }
                }
            }
        }
    } //

    public static void meanCpu(float[] x, int batch, int filters, int spatial, float[] mean, int oX, int oM) {

        float scale = 1.0f/(batch * spatial);

        for(int i = 0; i < filters; ++i){
            mean[oM + i] = 0;
            for(int j = 0; j < batch; ++j){
                for(int k = 0; k < spatial; ++k){
                    int index = j*filters*spatial + i*spatial + k;
                    mean[oM + i] += x[oX + index];
                }
            }
            mean[oM + i] *= scale;
        }
    } //

    public static void varianceCpu(float[] x, float[] mean, int batch, int filters, int spatial, float[] variance, int oX, int oM, int oV) {

        float scale = 1.0f/(batch * spatial - 1);

        for(int i = 0; i < filters; ++i){
            variance[oV + i] = 0;
            for(int j = 0; j < batch; ++j){
                for(int k = 0; k < spatial; ++k){
                    int index = j*filters*spatial + i*spatial + k;
                    variance[oV + i] += (float) Math.pow((x[oX + index] - mean[oM + i]), 2);
                }
            }
            variance[oV + i] *= scale;
        }
    } //

    public static void l2normalizeCpu(float[] x, float[] dx, int batch, int filters, int spatial, int oX, int oDx) {

        for(int b = 0; b < batch; ++b){
            for(int i = 0; i < spatial; ++i){
                float sum = 0;
                for(int f = 0; f < filters; ++f){
                    int index = b*filters*spatial + f*spatial + i;
                    sum += (float) Math.pow(x[oX + index], 2);
                }

                sum = (float) Math.sqrt(sum);

                for(int f = 0; f < filters; ++f){
                    int index = b*filters*spatial + f*spatial + i;
                    x[oX + index] /= sum;
                    dx[oDx + index] = (1 - x[oX + index]) / sum;
                }
            }
        }
    } //

    public static void normalizeCpu(float[] x, float[] mean, float[] variance, int batch, int filters, int spatial, int oX, int oM, int oV) {

        for(int b = 0; b < batch; ++b){
            for(int f = 0; f < filters; ++f){
                for(int i = 0; i < spatial; ++i){
                    int index = b*filters*spatial + f*spatial + i;
                    x[oX + index] = (x[oX + index] - mean[oM + f])/(float)(Math.sqrt(variance[oV + f]) + 0.000001f);
                }
            }
        }
    } //

    public static void constCpu(int N, float ALPHA, float[] X, int INCX, int oX) {

        for(int i = 0; i < N; ++i) X[oX + i*INCX] = ALPHA;
    } //

    public static void mulCpu(int N, float[] X, int INCX, float[] Y, int INCY, int oX, int oY) {

        for(int i = 0; i < N; ++i) Y[oY + i*INCY] *= X[oX + i*INCX];
    } //

    public static void powCpu(int N, float ALPHA, float[] X, int INCX, float[] Y, int INCY, int oX, int oY) {

        for(int i = 0; i < N; ++i) Y[oY + i*INCY] = (float) Math.pow(X[oX + i*INCX], ALPHA);
    } //

    public static void axpyCpu(int N, float ALPHA, float[] X, int INCX, float[] Y, int INCY, int oX, int oY) {

        for(int i = 0; i < N; ++i) Y[oY + i*INCY] += ALPHA*X[oX + i*INCX];
    } //

    public static void scalCpu(int N, float ALPHA, float[] X, int INCX, int oX) {

        for(int i = 0; i < N; ++i) X[oX + i*INCX] *= ALPHA;
    } //

    public static void fillCpu(int N, float ALPHA, float[] X, int INCX, int oX) {

        for(int i = 0; i < N; ++i) X[oX + i*INCX] = ALPHA;
    } //

    public static void deinterCpu(int NX, float[] X, int NY, float[] Y, int B, float[] OUT, int oX, int oY, int oOut) {

        int index = 0;

        for(int j = 0; j < B; ++j) {
            for(int i = 0; i < NX; ++i){

                if(X != null) {
                    X[oX + j*NX + i] += OUT[oOut + index];
                }
                ++index;
            }

            for(int i = 0; i < NY; ++i){

                if(Y != null) {
                    Y[oY + j*NY + i] += OUT[oOut + index];
                }
                ++index;
            }
        }
    } //

    public static void interCpu(int NX, float[] X, int NY, float[] Y, int B, float[] OUT, int oX, int oY, int oOut) {

        int index = 0;

        for(int j = 0; j < B; ++j) {
            for(int i = 0; i < NX; ++i){

                OUT[oOut + index++] = X[oX + j*NX + i];
            }

            for(int i = 0; i < NY; ++i){

                OUT[oOut + index++] = Y[oY + j*NY + i];
            }
        }
    } //

    public static void copyCpu(int N, float[] X, int INCX, float[] Y, int INCY, int oX, int oY) {

        for(int i = 0; i < N; ++i) Y[oY + i*INCY] = X[oX + i*INCX];
    } //

    public static void multAddIntoCpu(int N, float[] X, float[] Y, float[] Z, int oX, int oY, int oZ) {

        for(int i = 0; i < N; ++i) {
            Z[oZ + i] += X[oX + i] * Y[oY + i];
        }
    } //

    public static void smoothL1Cpu(int n, float[] pred, float[] truth, float[] delta, float[] error, int oP, int oT, int oD, int oE) {

        for(int i = 0; i < n; ++i){
            float diff = truth[oT + i] - pred[oP + i];
            float abs_val = Math.abs(diff);
            if(abs_val < 1) {
                error[oE + i] = diff * diff;
                delta[oD + i] = diff;
            }
            else {
                error[oE + i] = 2 * abs_val - 1;
                delta[oD + i] = (diff < 0) ? 1 : -1;
            }
        }
    } //

    public static void l1Cpu(int n, float[] pred, float[] truth, float[] delta, float[] error, int oP, int oT, int oD, int oE) {

        for(int i = 0; i < n; ++i){
            float diff = truth[oT + i] - pred[oP + i];
            error[oE + i] = Math.abs(diff);
            delta[oD + i] = (diff > 0) ? 1 : -1;
        }
    } //

    public static void softmaxXEntCpu(int n, float[] pred, float[] truth, float[] delta, float[] error, int oP, int oT, int oD, int oE) {

        for(int i = 0; i < n; ++i){
            float t = truth[oT + i];
            float p = pred[oP +i];
            error[oE + i] = (t != 0) ? - (float)Math.log(p) : 0;
            delta[oD + i] = t - p;
        }
    } //

    public static void logisticXEntCpu(int n, float[] pred, float[] truth, float[] delta, float[] error, int oP, int oT, int oD, int oE) {

        for(int i = 0; i < n; ++i){
            float t = truth[oT + i];
            float p = pred[oP + i];
            error[oE + i] = -1*(float)(t*Math.log(p) + (1-t)*Math.log(1-p));
            delta[oD + i] = t - p;
        }
    } //

    public static void l2Cpu(int n, float[] pred, float[] truth, float[] delta, float[] error, int oP, int oT, int oD, int oE) {

        for(int i = 0; i < n; ++i){
            float diff = truth[oT + i] - pred[oP + i];
            error[oE + i] = diff * diff;
            delta[oD + i] = diff;
        }
    } //

    public static float dotCpu(int N, float[] X, int INCX, float[] Y, int INCY, int oX, int oY) {

        float dot = 0;

        for(int i = 0; i < N; ++i) {
            dot += X[oX + i*INCX] * Y[oY + i*INCY];
        }
        return dot;
    } //

    public static void softmax(float[] input, int n, float temp, int stride, float[] output, int oIn, int oOut) {

        float sum = 0;
        float largest = - Rand.MAX_FLOAT;

        for(int i = 0; i < n; ++i){
            if(input[oIn + i*stride] > largest)  {
                largest = input[oIn + i*stride];
            }
        }

        for(int i = 0; i < n; ++i){
            float e = (float) Math.exp(input[oIn + i * stride] / temp - largest / temp);
            sum += e;
            output[oOut + i*stride] = e;
        }

        for(int i = 0; i < n; ++i){
            output[oOut + i*stride] /= sum;
        }
    } //

    public static void softmaxCpu(float[] input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float[] output, int oIn, int oOut) {

        for(int b = 0; b < batch; ++b){
            for(int g = 0; g < groups; ++g){

                int off = b*batch_offset + g*group_offset;

                softmax(input, n, temp, stride, output,off,off);

            }
        }
    } //

    public static void upsampleCpu(float[] in, int w, int h, int c, int batch, int stride, int forward, float scale, float[] out, int oIn, int oOut) {

        for(int b = 0; b < batch; ++b){
            for(int k = 0; k < c; ++k){
                for(int j = 0; j < h*stride; ++j){
                    for(int i = 0; i < w*stride; ++i){
                        int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                        int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;

                        if(forward != 0) {
                            out[oOut + out_index] = scale*in[oIn + in_index];
                        }
                        else {
                            in[oIn + in_index] += scale*out[oOut + out_index];
                        }
                    }
                }
            }
        }
    } //
}
