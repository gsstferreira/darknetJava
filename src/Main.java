import Server.Handlers.RequestHandler;
import Yolo.Setup;

import java.io.IOException;
import java.net.ServerSocket;
import java.util.Locale;

public class Main {

    private static final int port = 8080;

    public static void main(String[] args) {

        Locale.setDefault(Locale.ENGLISH);

        Setup.initYolo();
        System.gc();

        try {
            System.out.printf("Setting up server at port %d...\n",port);
            ServerSocket serverConnect = new ServerSocket(port);
            System.out.printf("Server ready, listening to port %d.\n",port);

            //noinspection InfiniteLoopStatement
            while(true) {
                new RequestHandler(serverConnect.accept());
            }

        }
        catch (IOException e) {
            System.err.println("Server setup error : " + e.getMessage());
        }
    }
}
