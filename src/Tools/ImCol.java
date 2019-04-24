package Tools;


import Classes.Arrays.FloatArray;

import java.util.stream.IntStream;

public abstract class ImCol {

    private static void col2ImAddPixel(FloatArray im, int height, int width, int channels, int row, int col, int channel, int pad, float val) {

        row -= pad;
        col -= pad;

        if(row >= 0 && col >= 0 && row < height && col < height) {

            int index = col + width*(row + height*channel);

            im.put(index, im.get(index) + val);
        }
    }

    private static float im2ColGetPixel(FloatArray im, int height, int width, int channels, int row, int col, int channel, int pad) {

        row -= pad;
        col -= pad;

        if(row < 0 || col < 0 || row >= height || col >= width) {
            return 0;
        }
        else {
            return im.get(col + width*(row + height*channel));
        }
    }

    public static void col2ImCpu(FloatArray dataCol, int channels, int height, int width, int ksize, int stride, int pad, FloatArray dataIm) {

        final int heightCol = (height + 2*pad - ksize) / stride + 1;
        final int widthCol = (width + 2*pad - ksize) / stride + 1;
        final int channelsCol = channels * ksize * ksize;

        IntStream.range(0,channelsCol).parallel().forEach(c -> {
            final int wOffset = c % ksize;
            final int hOffset = (c / ksize) % ksize;
            final int cIm = c / ksize / ksize;

            for (int h = 0; h < heightCol; ++h) {
                for (int w = 0; w < widthCol; ++w) {
                    int imRow = hOffset + h * stride;
                    int imCol = wOffset + w * stride;
                    int colIndex = (c * heightCol + h) * widthCol + w;
                    float val = dataCol.get(colIndex);
                    col2ImAddPixel(dataIm, height, width, channels,imRow, imCol, cIm, pad, val);
                }
            }
        });

//        for (int c = 0; c < channelsCol; ++c) {
//            int wOffset = c % ksize;
//            int hOffset = (c / ksize) % ksize;
//            int cIm = c / ksize / ksize;
//            for (int h = 0; h < heightCol; ++h) {
//                for (int w = 0; w < widthCol; ++w) {
//                    int imRow = hOffset + h * stride;
//                    int imCol = wOffset + w * stride;
//                    int colIndex = (c * heightCol + h) * widthCol + w;
//                    float val = dataCol.get(colIndex);
//                    col2ImAddPixel(dataIm, height, width, channels,imRow, imCol, cIm, pad, val);
//                }
//            }
//        }
    }

    public static void im2ColCpu(FloatArray dataIm, int channels, int height, int width, int ksize, int stride, int pad, FloatArray dataCol) {

        final int heightCol = (height + 2*pad - ksize) / stride + 1;
        final int widthCol = (width + 2*pad - ksize) / stride + 1;
        final int channelsCol = channels * ksize * ksize;

        IntStream.range(0,channelsCol).parallel().forEach( c -> {
            final int wOffset = c % ksize;
            final int hOffset = (c / ksize) % ksize;
            final int cIm = c / (ksize * ksize);

            for (int h = 0; h < heightCol; ++h) {

                final int imRow = hOffset + h * stride;
                final int colBase = (c * heightCol + h);

                for (int w = 0; w < widthCol; ++w) {

                    int imCol = wOffset + w * stride;
                    int colIndex = colBase * widthCol + w;
                    float val = im2ColGetPixel(dataIm, height, width, channels, imRow, imCol, cIm, pad);
                    dataCol.put(colIndex,val);
                }
            }
        });

//        for (int c = 0; c < channels_col; ++c) {
//            int wOffset = c % ksize;
//            int hOffset = (c / ksize) % ksize;
//            int cIm = c / ksize / ksize;
//            for (int h = 0; h < height_col; ++h) {
//                for (int w = 0; w < width_col; ++w) {
//                    int imRow = hOffset + h * stride;
//                    int imCol = wOffset + w * stride;
//                    int colIndex = (c * heightCol + h) * widthCol + w;
//                    float val = im2ColGetPixel(dataIm, height, width, channels, imRow, imCol, cIm, pad);
//                    dataCol.put(colIndex,val);
//                }
//            }
//        }
    }


}
