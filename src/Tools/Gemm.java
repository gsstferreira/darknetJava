package Tools;

import java.util.Random;
import java.util.stream.IntStream;

// Completo
public abstract class Gemm {

    public static void gemmBin(int M, int N, int K, float ALPHA, char[] A, int offA, int lda, float[] B, int offB, int ldb, float[] C, int offC, int ldc) {
        int i,j,k;
        for(i = 0; i < M; ++i){
            for(k = 0; k < K; ++k){
                char A_PART = A[offA + i*lda+k];
                if(A_PART != 0){
                    for(j = 0; j < N; ++j){
                        C[offC + i*ldc+j] += B[offB + k*ldb+j];
                    }
                } else {
                    for(j = 0; j < N; ++j){
                        C[offC + i*ldc+j] -= B[offB + k*ldb+j];
                    }
                }
            }
        }
    }

    public static float[] randomMatrix(int rows, int cols) {
        Random r = Rand.rand;

        int i;
        float[] m = new float[rows*cols];

        for(i = 0; i < rows*cols; ++i){
            m[i] = (1.0f*r.nextInt())/Rand.MAX_INT;
        }
        return m;
    }

    public static void timeRandomMatrix(int TA, int TB, int m, int k, int n) {
        float[] a;

        if(TA == 0) {
            a = randomMatrix(m,k);
        }
        else {
            a = randomMatrix(k,m);
        }

        int lda = (TA == 0)?k:m;
        float[] b;

        if(TB == 0) {
            b = randomMatrix(k,n);
        }
        else {
            b = randomMatrix(n,k);
        }
        int ldb = (TB == 0)?n:k;

        float[] c = randomMatrix(m,n);
        int i;

        long start = Util.DATE.getTime();
        for(i = 0; i<10; ++i){
            gemmCpu(TA,TB,m,n,k,1,a,0,lda,b,0,ldb,1,c,0,n);
        }
        long end = Util.DATE.getTime();

        System.out.print(String.format("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %d ms\n",m,k,k,n, TA, TB, end-start));
    }

    public static void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float[] A, int offA,  int lda, float[] B, int offB, int ldb, float BETA, float[] C, int offC, int ldc) {
        gemmCpu(TA,TB,M,N,K,ALPHA,A,offA,lda,B,offB,ldb,BETA,C,offC,ldc);
    }

    private static void gemmCpu(int TA, int TB, int M, int N, int K, float ALPHA, float[] A, int offA, int lda, float[] B, int offB, int ldb, float BETA, float[] C, int offC, int ldc) {
        int i, j;
        for(i = 0; i < M; ++i){
            for(j = 0; j < N; ++j){
                C[i*ldc + j] *= BETA;
            }
        }

        if(TA == 0) {
            if(TB == 0) {
                gemmNN(M, N, K, ALPHA,A,lda, B, ldb,C,ldc,offA,offB,offC);
            }
            else {
                gemmNT(M, N, K, ALPHA,A,lda, B, ldb,C,ldc,offA,offB,offC);
            }
        }
        else {
            if(TB == 0) {
                gemmTN(M, N, K, ALPHA,A,lda, B, ldb,C,ldc,offA,offB,offC);
            }
            else {
                gemmTT(M, N, K, ALPHA,A,lda, B, ldb,C,ldc,offA,offB,offC);
            }
        }
    }

    private static void gemmNN(int M, int N, int K, float ALPHA, float[] A, int lda, float[] B, int ldb, float[] C, int ldc, int offA, int offB, int offC) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for (int k = 0; k < K; ++k) {

                float A_PART = ALPHA * A[offA + i * lda + k];

                for (int j = 0; j < N; ++j) {
                    C[offC + i * ldc + j] += A_PART * B[offB + k * ldb + j];
                }
            }
        });
    }

    private static void gemmNT(int M, int N, int K, float ALPHA, float[] A, int lda, float[] B, int ldb, float[] C, int ldc, int offA, int offB, int offC) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for(int j = 0; j < N; ++j){

                float sum = 0;

                for(int k = 0; k < K; ++k){
                    sum += ALPHA * A[offA + i * lda + k] * B[offB + j * ldb + k];
                }
                C[offC + i * ldc + j] += sum;
            }
        });
    }

    private static void gemmTN(int M, int N, int K, float ALPHA, float[] A, int lda, float[] B, int ldb, float[] C, int ldc, int offA, int offB, int offC) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for (int k = 0; k < K; ++k) {

                float A_PART = ALPHA * A[offA + k * lda + i];

                for (int j = 0; j < N; ++j) {
                    C[offC + i * ldc + j] += A_PART * B[offB + k * ldb + j];
                }
            }
        });
    }

    private static void gemmTT(int M, int N, int K, float ALPHA, float[] A, int lda, float[] B, int ldb, float[] C, int ldc, int offA, int offB, int offC) {

        IntStream.range(0, M).parallel().forEach(i -> {

            for(int j = 0; j < N; ++j){

                float sum = 0;

                for(int k = 0; k < K; ++k){
                    sum += ALPHA * A[offA + i + k * lda] * B[offB + k + j * ldb];
                }
                C[offC + i * ldc + j] += sum;
            }
        });
    }

}
