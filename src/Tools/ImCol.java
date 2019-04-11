package Tools;


import Classes.Buffers.FloatBuffer;

public abstract class ImCol {

    private static void col2ImAddPixel(FloatBuffer im, int height, int width, int channels, int row, int col, int channel, int pad, float val) {

        row -= pad;
        col -= pad;

        if(row >= 0 && col >= 0 && row < height && col < height) {

            int index = col + width*(row + height*channel);

            im.put(index, im.get(index) + val);
        }
    }

    private static float im2ColGetPixel(FloatBuffer im,int height, int width, int channels, int row, int col, int channel, int pad) {

        row -= pad;
        col -= pad;

        if(row < 0 || col < 0 || row >= height || col >= width) {
            return 0;
        }
        else {
            return im.get(col + width*(row + height*channel));
        }
    }

    public static void col2ImCpu(FloatBuffer dataCol, int channels, int height, int width, int ksize, int stride, int pad, FloatBuffer dataIm) {

        int c,h,w;
        int height_col = (height + 2*pad - ksize) / stride + 1;
        int width_col = (width + 2*pad - ksize) / stride + 1;

        int channels_col = channels * ksize * ksize;
        for (c = 0; c < channels_col; ++c) {
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = 0; h < height_col; ++h) {
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    float val = dataCol.get(col_index);
                    col2ImAddPixel(dataIm, height, width, channels,im_row, im_col, c_im, pad, val);
                }
            }
        }
    }

    public static void im2ColCpu(FloatBuffer dataIm, int channels, int height, int width, int ksize, int stride, int pad, FloatBuffer dataCol) {

        int c,h,w;
        int height_col = (height + 2*pad - ksize) / stride + 1;
        int width_col = (width + 2*pad - ksize) / stride + 1;

        int channels_col = channels * ksize * ksize;
        for (c = 0; c < channels_col; ++c) {
            int w_offset = c % ksize;
            int h_offset = (c / ksize) % ksize;
            int c_im = c / ksize / ksize;
            for (h = 0; h < height_col; ++h) {
                for (w = 0; w < width_col; ++w) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    float val = im2ColGetPixel(dataIm, height, width, channels, im_row, im_col, c_im, pad);
                    dataCol.put(col_index,val);
                }
            }
        }
    }


}
