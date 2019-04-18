
import Server.Handlers.RequestHandler;
import Yolo.Setup;

import java.io.*;
import java.net.ServerSocket;

public class Main {

    private static final int port = 8080;

    public static void main(String[] args) {

        Setup.initYolo();

        try {
            ServerSocket serverConnect = new ServerSocket(port);
            System.out.printf("Server ready, listening port %d.\n",port);

            while(true) {

                new RequestHandler(serverConnect.accept());
            }

        } catch (IOException e) {
            System.err.println("Server Connection error : " + e.getMessage());
        }
    }
}
