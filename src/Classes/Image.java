package Classes;

import Enums.ImType;
import Tools.Blas;
import Tools.Rand;
import Tools.Util;
import org.lwjgl.BufferUtils;
import org.lwjgl.stb.STBImage;
import org.lwjgl.stb.STBImageWrite;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

public class Image {

    public int w;
    public int h;
    public int c;
    public FloatBuffer data;

    private static int windows = 0;
    private static float[][] colors = {{1,0,1},{0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0}};

    private static final int alphabetNsize = 8;

    public Image(int width, int height, int c, boolean random) {
        this.w = width;
        this.h = height;
        this.c = c;

        this.data = BufferUtils.createFloatBuffer(width*height*c);

        if(random) {
            for(int i = 0; i < w*h*c; ++i){

                double val = (Rand.randNormal() * 0.25) + 0.5;
                this.data.put(i,(float)val);
            }
        }
    }

    public Image(int width, int height, int c, FloatBuffer data) {

        this.w = width;
        this.h = height;
        this.c = c;

        this.data = BufferUtils.createFloatBuffer(width*height*c);

        for(int i = 0; i < this.data.capacity(); i++) {
                this.data.put(i,data.get(i));
        }
    }

    public static float getColor(int c, int x, int max) {

        float ratio = ((float)x/max)*5;
        int i = (int) Math.floor(ratio);
        int j = (int) Math.ceil(ratio);
        ratio -= i;
        float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
        return r;
    }

    public Image maskToRgb() {

        int n = this.c;
        Image im = new Image(w,h,3,false);
        int i, j;
        for(j = 0; j < n; ++j){
            int offset = j*123457 % n;
            float red = getColor(2,offset,n);
            float green = getColor(1,offset,n);
            float blue = getColor(0,offset,n);
            for(i = 0; i < im.w*im.h; ++i){

                int index = im.w*im.h;

                float val1 = im.data.get(i) + this.data.get(j*index + i) * red;
                float val2 = im.data.get(i + index) + this.data.get(j*index + i) * green;
                float val3 = im.data.get(i + 2*index) + this.data.get(j*index + i) * blue;

                im.data.put(i,val1);
                im.data.put(i + index, val2);
                im.data.put(i + 2*index, val3);
            }
        }
        return im;
    }

    public float getPixel(int x, int y, int c) {

        try {
            if(x >= w || y >= h || c >= this.c || x < 0 || y < 0) {
                throw new Exception();
            }

            return this.data.get(c*h*w + y*w + x);
        }
        catch (Exception e) {
            System.out.println("Image.getPixel() coords ou of bounds.");
            return Float.NaN;
        }
    }

    public float getPixelExtended(int x, int y, int c) {

        float val = this.getPixel(x,y,c);

        if(Float.isNaN(val)) {
            return 0;
        }
        else {
            return val;
        }
    }

    public void setPixel(int x, int y, int c, float val) {

        try {

            if(x >= w || y >= h || c >= this.c || x < 0 || y < 0){
                return;
            }

            this.data.put(c*h*w + y*w + x,val);
        }
        catch (Exception e) {
            System.out.println("Image.setPixel() coords ou of bounds.");
        }
    }

    public void addToPixel(int x, int y, int c, float val) {

        float v = getPixel(x,y,c);
        setPixel(x,y,c,val + v);
    }

    public float bilinearInterpolate(float x, float y, int c) {

        int ix = (int) Math.floor(x);
        int iy = (int) Math.floor(y);

        float dx = x - ix;
        float dy = y - iy;

        float val = (1-dy) * (1-dx) * getPixelExtended(ix,iy,c);
        val += dy * (1-dx) * getPixelExtended(ix,iy+1,c);
        val += (1-dy) * dx * getPixelExtended(ix+1,iy,c);
        val += dy * dx * getPixelExtended(ix+1,iy+1,c);

        return val;
    }

    public void compositeImage(Image dest, int dx, int dy) {

        int x,y,k;
        for(k = 0; k < c; ++k){
            for(y = 0; y < h; ++y){
                for(x = 0; x < w; ++x){
                    float val = this.getPixel(x,y,k);
                    float val2 = dest.getPixelExtended(dx+x, dy+y, k);
                    dest.setPixel(dx+x, dy+y, k, val * val2);
                }
            }
        }
    }
    
    public Image borderImage(int border) {

        Image b = new Image(w + 2*border, h + 2*border, c,false);
        int x,y,k;
        for(k = 0; k < b.c; ++k){
            for(y = 0; y < b.h; ++y){
                for(x = 0; x < b.w; ++x){
                    float val = this.getPixelExtended(x - border, y - border, k);
                    if(x - border < 0 || x - border >= w || y - border < 0 || y - border >= h) {
                        val = 1;
                    }
                    b.setPixel(x, y, k, val);
                }
            }
        }
        return b;
    }

    public Image tileImages(Image b, int dx) {

        if(w == 0) {
            return b.copyImage();
        }
        else {
            Image c = new Image(w + b.w + dx,(h > b.h) ? h : b.h, (this.c > b.c) ? this.c : b.c,false);
            Blas.fillCpu(c.w*c.h*c.c, 1, c.data, 1);
            this.embedImage(c, 0, 0);
            b.compositeImage(c, w + dx, 0);
            return c;
        }
    }

    public Image getLabel(Image[][] characters, String string, int size) {

        size = size/10;
        if(size > 7) size = 7;
        Image label = new Image(0,0,0,false);


        for (int i = 0; i < string.length(); i++) {

            Image l = characters[size][string.charAt(i)];
            label = label.tileImages(l,-size - 1 + (size + 1)/2);
        }
        return label.borderImage(label.h/4);
    }

    public void embedImage(Image dest, int dx, int dy) {

        int x,y,k;
        for(k = 0; k < c; ++k){
            for(y = 0; y < h; ++y){
                for(x = 0; x < w; ++x){
                    float val = getPixel(x,y,k);
                    dest.setPixel(dx+x, dy+y, k, val);
                }
            }
        }
    }

    public Image copyImage() {

        Image copy = new Image(w,h,c,false);

        for(int i = 0; i < data.capacity(); i++) {
            copy.data.put(i,data.get(i));
        }
        return copy;
    }

    public Image resizeImage(int w, int h) {

        Image resized = new Image(w, h, this.c,false);
        Image part = new Image(w, this.h, this.c,false);

        int r, c, k;
        float w_scale = (float)(this.w - 1) / (w - 1);
        float h_scale = (float)(this.h - 1) / (h - 1);
        for(k = 0; k < this.c; ++k){
            for(r = 0; r < this.h; ++r){
                for(c = 0; c < w; ++c){
                    float val = 0;
                    if(c == w-1 || this.w == 1){
                        val = getPixel(this.w-1, r, k);
                    } else {
                        float sx = c*w_scale;
                        int ix = (int) sx;
                        float dx = sx - ix;
                        val = (1 - dx) * getPixel(ix, r, k) + dx * getPixel(ix+1, r, k);
                    }
                    part.setPixel(c, r, k, val);
                }
            }
        }
        for(k = 0; k < this.c; ++k){
            for(r = 0; r < h; ++r){
                float sy = r*h_scale;
                int iy = (int) sy;
                float dy = sy - iy;
                for(c = 0; c < w; ++c){
                    float val = (1-dy) * part.getPixel(c, iy, k);
                    resized.setPixel(c, r, k, val);
                }
                if(r == h-1 || this.h == 1) continue;
                for(c = 0; c < w; ++c){
                    float val = dy * part.getPixel(c, iy + 1, k);
                    resized.addToPixel(c, r, k, val);
                }
            }
        }
        return resized;
    }

    public void drawLabel(int r, int c, Image label, FloatBuffer rgb) {

        int _w = label.w;
        int _h = label.h;
        if (r - _h >= 0) r = r - _h;

        int i, j, k;
        for(j = 0; j < _h && j + r < h; ++j){
            for(i = 0; i < _w && i + c < w; ++i){
                for(k = 0; k < label.c; ++k){
                    float val = getPixel(i, j, k);
                    this.setPixel(i+c, j+r, k, rgb.get(k) * val);
                }
            }
        }
    }

    public void drawBox(int x1, int y1, int x2, int y2, float r, float g, float b) {

        int i;
        if(x1 < 0) x1 = 0;
        if(x1 >= w) x1 = w-1;
        if(x2 < 0) x2 = 0;
        if(x2 >= w) x2 = w-1;

        if(y1 < 0) y1 = 0;
        if(y1 >= h) y1 = h-1;
        if(y2 < 0) y2 = 0;
        if(y2 >= h) y2 = h-1;

        for(i = x1; i <= x2; ++i) {

            data.put(i + y1*w,r);
            data.put(i + y2*w,r);

            data.put(i + (y1 + h)*w,g);
            data.put(i + (y2 + h)*w,g);

            data.put(i + (y1 + 2*h)*w,b);
            data.put(i + (y2 + 2*h)*w,b);
        }
        for(i = y1; i <= y2; ++i) {

            data.put(x1 + i*w,r);
            data.put(x2 + i*w,r);

            data.put(x1 + i*w + w*h,g);
            data.put(x2 + i*w + w*h,g);

            data.put(x1 + i*w + 2*w*h,b);
            data.put(x2 + i*w + 2*w*h,b);
        }
    }

    public void drawBoxWidth(int x1, int y1, int x2, int y2, int w, float r, float g, float b) {

        int i;
        for(i = 0; i < w; ++i){
            drawBox(x1+i, y1+i, x2-i, y2-i, r, g, b);
        }
    }

    public void drawBbox(Box bbox, int w, float r, float g, float b) {

        int left  = (int) (bbox.x-bbox.w/2)*this.w;
        int right = (int) (bbox.x+bbox.w/2)*this.w;
        int top   = (int) (bbox.y-bbox.h/2)*this.h;
        int bot   = (int) (bbox.y+bbox.h/2)*this.h;

        int i;
        for(i = 0; i < w; ++i){
            drawBox(left+i, top+i, right-i, bot-i, r, g, b);
        }
    }

    public static Image loadImageColor(String fileName,int w, int h) {
        return loadImage(fileName,w,h,3);
    }

    public static Image loadImage(String filename, int w, int h, int c) {

        Image out = loadImageStb(filename, c);

        if((h != 0 && w != 0) && (h != out.h || w != out.w)){

            out = out.resizeImage(w, h);
        }
        return out;
    }

    public static Image loadImageStb(String filename, int channels) {

        //Emulando ponteiros
        int[] _width = new int[1];
        int[] _height = new int[1];
        int[] _channel = new int[1];

        try {

            ByteBuffer bb = STBImage.stbi_load(filename,_width,_height,_channel,channels);

            int width = _width[0];
            int height = _height[0];
            int channel = (channels != 0) ? channels : _channel[0];

            Image im = new Image(width, height, channels,false);
            for(int k = 0; k < channel; ++k){
                for(int j = 0; j < height; ++j){
                    for(int i = 0; i < width; ++i){
                        int dst_index = i + width*j + width*height*k;
                        int src_index = k + channel*i + channel*width*j;
                        im.data.put(dst_index,bb.get(src_index)/255.0f);
                    }
                }
            }
            return im;
        }
        catch (Exception e) {
            System.out.println(String.format("Cannot load Image '%s'",filename));
            e.printStackTrace();
            return null;
        }
    }

    private void saveImageOptions(String name, ImType f, int quality) {

        String s;

        switch (f) {
            case BMP:
                s = String.format("%s.bmp",name);
                break;
            case JPG:
                s = String.format("%s.jpg",name);
                break;
            case TGA:
                s = String.format("%s.tga",name);
                break;
            default:
                s = String.format("%s.png",name);
                break;
        }

        byte[] data = new byte[w*h*c];

        int i,k;
        for(k = 0; k < c; ++k){
            for(i = 0; i < w*h; ++i){
                data[i*c+k] = (byte) (255*this.data.get(i+k*w*h));
            }
        }

        ByteBuffer bb = BufferUtils.createByteBuffer(data.length);
        bb = bb.put(data,0,data.length);
        bb.position(0);

        boolean success;

        switch (f) {
            case BMP:
                success = STBImageWrite.stbi_write_bmp(s,w,h,c,bb);
                break;
            case JPG:
                success = STBImageWrite.stbi_write_jpg(s,w,h,c,bb,quality);
                break;
            case TGA:
                success = STBImageWrite.stbi_write_tga(s,w,h,c,bb);
                break;
            default:
                success = STBImageWrite.stbi_write_png(s,w,h,c,bb,w*c);
                break;
        }
        if(!success) {
            System.out.println(String.format("Cannot save Image Image '%s'",s));
        }
    }

    public void saveToDisk(String path, ImType f, int quality) {

        this.saveImageOptions(path,f,quality);
    }

    public static Image[][] loadAlphabet() {

        Image[][] alphabets = new Image[alphabetNsize][128];

        for(int j = 0; j < alphabetNsize; ++j){
            for(int i = 32; i < 127; ++i){
                String s = String.format("data/labels/%d_%d.png",i,j);
                alphabets[j][i] = loadImageColor(s, 0, 0);
            }
        }
        return alphabets;
    }

    public Image getImageLayer(int l) {
        Image out = new Image(w,h,1,false);

        for(int i = 0; i < w*h; i++) {
            out.data.put(i,this.data.get(i+l*h*w));
        }
        return out;
    }

    public static Image collapseImagesVert(Image[] ims, int n) {

        int border = 1;
        int w = ims[0].w;
        int h = (ims[0].h + border) * n - border;
        int c = ims[0].c;

        if(c != 3) {
            w = (w+border)*c - border;
            c = 1;
        }

        Image filters = new Image(w,h,c,false);

        for(int i = 0; i < n; ++i){
            int h_offset = i*(ims[0].h+border);
            Image copy = ims[i].copyImage();

            if(c == 3){
                copy.embedImage(filters, 0, h_offset);
            }
            else{
                for(int j = 0; j < copy.c; ++j){
                    int w_offset = j*(ims[0].w+border);
                    Image layer = copy.getImageLayer(j);
                    layer.embedImage(filters, w_offset, h_offset);
                }
            }
        }
        return filters;
    }

    public static Image collapseImagesHorz(Image[] ims, int n) {


        int color = 1;
        int border = 1;

        int size = ims[0].h;
        int h = size;
        int w = (ims[0].w + border) * n - border;
        int c = ims[0].c;

        if(c != 3){
            h = (h+border)*c - border;
            c = 1;
        }

        Image filters = new Image(w, h, c,false);

        for(int i = 0; i < n; ++i){
            int w_offset = i*(size+border);
            Image copy = ims[i].copyImage();

            if(c == 3){
                copy.embedImage(filters, w_offset, 0);
            }
            else{
                for(int j = 0; j < copy.c; ++j){
                    int h_offset = j*(size+border);
                    Image layer = copy.getImageLayer(j);
                    layer.embedImage(filters, w_offset, h_offset);
                }
            }
        }
        return filters;
    }
}
