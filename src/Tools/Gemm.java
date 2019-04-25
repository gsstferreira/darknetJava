package Tools;

import Classes.Arrays.FloatArray;
import java.util.stream.IntStream;

public abstract class Gemm {

//    public static void gemmBin(int M, int N, int K, float ALPHA, ByteArray A, int lda, FloatArray B, int ldb, FloatArray C, int ldc) {
//        int i,j,k;
//        for(i = 0; i < M; ++i){
//            for(k = 0; k < K; ++k){
//                byte A_PART = A.get(i*lda+k);
//                if(A_PART != 0){
//                    for(j = 0; j < N; ++j){
//
//                        float val = C.get(i*ldc+j) + B.get(k*ldb+j);
//                        C.set(i*ldc+j,val);
//                    }
//                } else {
//                    for(j = 0; j < N; ++j){
//                        float val = C.get(i*ldc+j) - B.get(k*ldb+j);
//                        C.set(i*ldc+j,val);
//                    }
//                }
//            }
//        }
//    }
//
//    public static FloatArray randomMatrix(int rows, int cols) {
//
//        int i;
//        float[] m = new float[rows*cols];
//
//        for(i = 0; i < rows*cols; ++i){
//            m[i] = (1.0f*Rand.randInt())/Rand.MAX_INT;
//        }
//        return new FloatArray(m);
//    }
//
//    public static void timeRandomMatrix(int TA, int TB, int m, int k, int n) {
//        FloatArray a;
//
//        if(TA == 0) {
//            a = randomMatrix(m,k);
//        }
//        else {
//            a = randomMatrix(k,m);
//        }
//
//        int lda = (TA == 0)?k:m;
//        FloatArray b;
//
//        if(TB == 0) {
//            b = randomMatrix(k,n);
//        }
//        else {
//            b = randomMatrix(n,k);
//        }
//        int ldb = (TB == 0)?n:k;
//
//        FloatArray c = randomMatrix(m,n);
//        int i;
//
//        long start = System.currentTimeMillis();
//        for(i = 0; i<10; ++i){
//            gemmCpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
//        }
//        long end = System.currentTimeMillis();
//
//        System.out.print(String.format("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %d ms\n",m,k,k,n, TA, TB, end-start));
//    }

    public static void gemm(int TA, int TB, int M, int N, int K, float ALPHA, FloatArray A, int lda, FloatArray B, int ldb, float BETA, FloatArray C, int ldc) {
        gemmCpu(TA,TB,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);
    }

    public static void gemmCpu(int TA, int TB, int M, int N, int K, float ALPHA, FloatArray A, int lda, FloatArray B, int ldb, float BETA, FloatArray C, int ldc) {

        IntStream.range(0,M).parallel().forEach(i -> {

            final int index = i *ldc;

            for(int j = 0; j < N; ++j){

                C.mulIn(index + j,BETA);
            }
        });

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

    private static void gemmNN(int M, int N, int K, float ALPHA, FloatArray A, int lda, FloatArray B, int ldb, FloatArray C, int ldc) {

        IntStream.range(0,M).parallel().forEach(i -> {

            final int iLda = i * lda;
            final int iLdc = i * ldc;

            for (int k = 0; k < K; ++k) {

                final float aPart = ALPHA * A.get(iLda +k);
                final int kLdb = k * ldb;

                for (int j = 0; j < N; ++j) {
                    C.addIn(iLdc + j,aPart*B.get(kLdb + j));
                }
            }
        });
    }

    private static void gemmNT(int M, int N, int K, float ALPHA, FloatArray A, int lda, FloatArray B, int ldb, FloatArray C, int ldc) {

        IntStream.range(0,M).parallel().forEach(i -> {

            final int iLda = i * lda;
            final int iLdc = i * ldc;

            for(int j = 0; j < N; ++j){

                float sum = 0;
                final int jLdb = j * ldb;

                for(int k = 0; k < K; ++k){
                    sum += ALPHA * A.get(iLda + k) * B.get(jLdb + k);
                }

                C.addIn(iLdc + j, sum);
            }
        });
    }

    private static void gemmTN(int M, int N, int K, float ALPHA, FloatArray A, int lda, FloatArray B, int ldb, FloatArray C, int ldc) {

        IntStream.range(0,M).parallel().forEach( i-> {

            final int iLdc = i * ldc;

            for (int k = 0; k < K; ++k) {

                final float aPart = ALPHA * A.get(k*lda +i);
                final int kLdb = k * ldb;

                for (int j = 0; j < N; ++j) {

                    C.addIn(iLdc + j,aPart*B.get(kLdb + j));
                }
            }
        });
    }

    private static void gemmTT(int M, int N, int K, float ALPHA, FloatArray A, int lda, FloatArray B, int ldb, FloatArray C, int ldc) {

        IntStream.range(0,M).parallel().forEach( i-> {

            final int iLdc = i * ldc;

            for(int j = 0; j < N; ++j){

                float sum = 0;
                final int jLdb = j * ldb;

                for(int k = 0; k < K; ++k){
                    sum += ALPHA * A.get(i+ lda*k) * B.get(jLdb + k);
                }

                C.addIn(iLdc + j,sum);
            }
        });
    }
}
