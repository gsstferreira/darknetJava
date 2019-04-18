
import Server.Handlers.RequestHandler;
import Yolo.Setup;
import org.lwjgl.system.CallbackI;

import java.io.*;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;

public class Main {

    private static final int port = 8080;

    public static void main(String[] args) {

        Setup.initYolo();
        System.gc();

        try {
            System.out.printf("Setting up server at port %d...\n",port);
            ServerSocket serverConnect = new ServerSocket(port);
            System.out.printf("Server ready, listening to port %d.\n",port);

            while(true) {

                new RequestHandler(serverConnect.accept());
            }

        } catch (IOException e) {
            System.err.println("Server setup error : " + e.getMessage());
        }
    }
}
