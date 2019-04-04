package Tools;

import java.nio.CharBuffer;
import java.nio.FloatBuffer;
import java.util.Random;
import java.util.stream.IntStream;

// Completo
public abstract class Gemm {

    public static void gemmBin(int M, int N, int K, float ALPHA, CharBuffer A, int lda, FloatBuffer B, int ldb, FloatBuffer C, int ldc) {
        int i,j,k;
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                char A_PART = A.get(i*lda+k);
                if(A_PART != 0){
                    for(j = 0; j < N; ++j){
                        
                        float val = C.get(i*ldc+j) + B.get(k*ldb+j);
                        C.put(i*ldc+j,val);
                    }
                } else {
                    for(j = 0; j < N; ++j){
                        float val = C.get(i*ldc+j) - B.get(k*ldb+j);
                        C.put(i*ldc+j,val);
                    }
                }
            }
        }
    }

    public static FloatBuffer randomMatrix(int rows, int cols) {

        int i;
        float[] m = new float[rows*cols];

        for(i = 0; i < rows*cols; ++i){
            m[i] = (1.0f*Rand.randInt())/Rand.MAX_INT;
        }
        return FloatBuffer.wrap(m);
    }

    public static void timeRandomMatrix(int TA, int TB, int m, int k, int n) {
        FloatBuffer a;

        if(TA == 0) {
            a = randomMatrix(m,k);
        }
        else {
            a = randomMatrix(k,m);
        }

        int lda = (TA == 0)?k:m;
        FloatBuffer b;

        if(TB == 0) {
            b = randomMatrix(k,n);
        }
        else {
            b = randomMatrix(n,k);
        }
        int ldb = (TB == 0)?n:k;

        FloatBuffer c = randomMatrix(m,n);
        int i;

        long start = Util.getTime();
        for(i = 0; i<10; ++i){
            gemmCpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
        }
        long end = Util.getTime();

        System.out.print(String.format("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %d ms\n",m,k,k,n, TA, TB, end-start));
    }

    public static void gemm(int TA, int TB, int M, int N, int K, float ALPHA, FloatBuffer A,  int lda, FloatBuffer B, int ldb, float BETA, FloatBuffer C, int ldc) {
        gemmCpu(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }

    public static void gemmCpu(int TA, int TB, int M, int N, int K, float ALPHA, FloatBuffer A, int lda, FloatBuffer B, int ldb, float BETA, FloatBuffer C, int ldc) {
        int i, j;
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){

                int index = i*ldc + j;
                C.put(index,C.get(index)*BETA);
            }
        }

        if(TA == 0) {
            if(TB == 0) {
                gemmNN(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
            }
            else {
                gemmNT(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
            }
        }
        else {
            if(TB == 0) {
                gemmTN(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
            }
            else {
                gemmTT(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
            }
        }
    }

    private static void gemmNN(int M, int N, int K, float ALPHA, FloatBuffer A, int lda, FloatBuffer B, int ldb, FloatBuffer C, int ldc) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for (int k = 0; k < K; ++k) {

                float A_PART = ALPHA * A.get(i*lda +k);

                for (int j = 0; j < N; ++j) {
                    float val = C.get(i*ldc+j) + A_PART*B.get(k*ldb+j);
                    C.put(i*ldc+j,val);
                }
            }
        });
    }

    private static void gemmNT(int M, int N, int K, float ALPHA, FloatBuffer A, int lda, FloatBuffer B, int ldb, FloatBuffer C, int ldc) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for(int j = 0; j < N; ++j){

                float sum = 0;

                for(int k = 0; k < K; ++k){
                    sum += ALPHA * A.get(i*lda+k) * B.get(j*ldb + k);
                }

                C.put(i*ldc+j,C.get(i*ldc+j) + sum);
            }
        });
    }

    private static void gemmTN(int M, int N, int K, float ALPHA, FloatBuffer A, int lda, FloatBuffer B, int ldb, FloatBuffer C, int ldc) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for (int k = 0; k < K; ++k) {

                float A_PART = ALPHA * A.get(k*lda +i);

                for (int j = 0; j < N; ++j) {

                    float val = C.get(i*ldc+j) + A_PART*B.get(k*ldb+j);
                    C.put(i*ldc+j,val);
                }
            }
        });
    }

    private static void gemmTT(int M, int N, int K, float ALPHA, FloatBuffer A, int lda, FloatBuffer B, int ldb, FloatBuffer C, int ldc) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for(int j = 0; j < N; ++j){

                float sum = 0;

                for(int k = 0; k < K; ++k){
                    sum += ALPHA * A.get(i+ lda*k) * B.get(j*ldb + k);
                }
                C.put(i*ldc+j,C.get(i*ldc+j) + sum);
            }
        });
    }

}
