package Classes;

import Tools.Rand;
import Tools.Util;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class Matrix {

    public int rows;
    public int cols;
    public float[][] vals;

    public Matrix(int rows, int cols) {

        this.rows = rows;
        this.cols = cols;

        this.vals = new float[rows][cols];
    }

    public static float matrixTopkAccuracy(Matrix truth, Matrix guess, int k) {

        int[] indexes = new int[k];
        int n = truth.cols;

        int correct = 0;

        for(int i = 0; i < truth.rows; i++) {

            IntBuffer ib = IntBuffer.wrap(indexes);
            FloatBuffer fb = FloatBuffer.wrap(guess.vals[i]);

            Util.topK(fb,n,k,ib);
            for(int j = 0; j < k; j++) {
                int _class = indexes[j];

                if(truth.vals[i][_class] != 0) {
                    correct++;
                    break;
                }
            }
        }
        return (1.0f*correct)/truth.rows;
    }

    public void scale(float scale) {
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                vals[i][j] *= scale;
            }
        }
    }

    public void resize(int size) {

        if(rows != size) {

            float[][] arr = new float[size][cols];
            System.arraycopy(vals,0,arr,0,size);

            vals = arr;
            rows = size;
        }
    }

    public void addTo(Matrix m) {

        if(rows == m.rows && cols == m.cols) {
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    vals[i][j] += m.vals[i][j];
                }
            }
        }
        else {
            System.out.println("Matrix::addTo() - cannot add matrixes of different sizes.");
        }
    }

    public Matrix copy() {

        Matrix m = new Matrix(rows,cols);

        m.addTo(this);

        return m;
    }

    public Matrix holdOutMatrix(int n) {

        Matrix h = new Matrix(n,cols);

        for(int i = 0; i < n; i++) {
            int index = Rand.rand.nextInt()%rows;
            h.vals[i] = vals[index];
            vals[index] = vals[--rows];
        }

        return h;
    }

    public float[] popColumn(int c) {

        float[] col = new float[rows];

        for(int i = 0; i < rows; i++) {
            col[i] = vals[i][c];
            for(int j = c; j < cols - 1; j++) {
                vals[i][j] = vals[i][j+1];
            }
        }
        cols--;
        return col;
    }

}
