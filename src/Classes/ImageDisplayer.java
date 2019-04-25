package Classes;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ImageDisplayer extends JFrame {

    public ImageDisplayer(String path, int width, int height) {
        setTitle("Last Prediction");
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        updateImage(path,width,height);
    }

    public void updateImage(String path, int width, int height) {

        setSize(width,height);
        getContentPane().removeAll();
        add(new ImageComponent(path));
        validate();
        repaint();
        setVisible(true);
    }

    private class ImageComponent extends JComponent{

        private static final long serialVersionUID = 1L;
        private BufferedImage image;

        ImageComponent(String path){
            try{
                File image2 = new File(path);
                image = ImageIO.read(image2);

            }
            catch (IOException e){
                System.err.println("Unable to display labeled image.");
            }
        }
        public void paintComponent (Graphics g){
            if(image == null) return;
            int imageWidth = image.getWidth(this);
            int imageHeight = image.getHeight(this);

            g.drawImage(image, 0, 0, this);

            for (int i = 0; i*imageWidth <= getWidth(); i++)
                for(int j = 0; j*imageHeight <= getHeight();j++)
                    if(i+j>0) g.copyArea(0, 0, imageWidth, imageHeight, i*imageWidth, j*imageHeight);
        }

    }
}
