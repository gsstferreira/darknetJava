
import Classes.DetectionResult;
import Tools.Detector;
import Tools.GlobalVars;
import Tools.Setup;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;
import java.util.StringTokenizer;

public class Main implements Runnable{

    private static final int port = 8080;
    private static final Gson gson = new Gson();

    private Socket connect;

    private Main(Socket c) {
        this.connect = c;
    }

    public static void main(String[] args) {

        Setup.initYolo();

        try {
            ServerSocket serverConnect = new ServerSocket(port);
            System.out.println("Server started.\nListening port is " + port + ".\n");

            // we listen until user halts server execution
            while (true) {

                Main myServer = new Main(serverConnect.accept());

                // create dedicated thread to manage the client connection
                Thread thread = new Thread(myServer);
                thread.start();
            }

        } catch (IOException e) {
            System.err.println("Server Connection error : " + e.getMessage());
        }
    }

    @Override
    public void run() {


        Date d = new Date();

        BufferedReader in = null;
        PrintWriter out = null;
        BufferedOutputStream dataOut = null;
        String requestParams;

        try {
            // we read characters from the client via input stream on the socket
            in = new BufferedReader(new InputStreamReader(connect.getInputStream()));
            // we get character output stream to client (for headers)
            out = new PrintWriter(connect.getOutputStream());
            // get binary output stream to client (for requested data)
            dataOut = new BufferedOutputStream(connect.getOutputStream());

            // get first line of the request from the client
            String input = in.readLine();
            // we parse the request with a string tokenizer
            StringTokenizer parse = new StringTokenizer(input);
            String method = parse.nextToken().toUpperCase(); // we get the HTTP method of the client
            // we get file requested
            requestParams = parse.nextToken();

            if(requestParams.contains("favicon.ico")) {
                return;
            }

            String[] ss = requestParams.split("/");
            String[] s1 = ss[1].split("\\?");

            if(!s1[0].equals("detect")) {
                throw new IOException("bad request");
            }

            System.out.println("Connecton opened. (" + d + ")");

            String[] s2 = s1[1].split("&");

            String imgPath = s2[0].split("=")[1];
            float thresh  = Float.parseFloat(s2[1].split("=")[1]);


            if (method.equals("GET")) { // GET method so we return content

                DetectionResult detections = Detector.runDetector(imgPath,thresh);
                String result = gson.toJson(detections);

                // send HTTP Headers
                out.println("HTTP/1.1 200 OK");
                out.println("Server: Java HTTP Server from SSaurel : 1.0");
                out.println("Date: " + new Date());
                out.println("Content-type: application/json");
                out.println("Content-length: " + result.length());
                out.println(); // blank line between headers and content, very important !
                out.flush(); // flush character output stream buffer

                dataOut.write(result.getBytes(), 0, result.length());
                dataOut.flush();
            }

        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.err.println("Server error : " + ioe);
        } finally {
            try {
                in.close();
                out.close();
                dataOut.close();
                connect.close(); // we close socket connection
            } catch (Exception e) {
                System.err.println("Error closing stream : " + e.getMessage());
            }
        }
    }
}
