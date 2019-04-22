package Server.Handlers;

import java.io.*;
import java.net.Socket;
import java.util.Date;
import java.util.StringTokenizer;

public class RequestHandler implements Runnable {

    private BufferedReader requestInput;
    private PrintWriter responseOutput;
    private BufferedOutputStream responseData;

    private Socket connection;

    public RequestHandler(Socket conn) {
        this.connection = conn;

        try {
            requestInput = new BufferedReader(new InputStreamReader(connection.getInputStream()));
            responseOutput = new PrintWriter(connection.getOutputStream());
            responseData = new BufferedOutputStream(connection.getOutputStream());
        }
        catch (IOException e) {
            System.err.println("Socket connection error: " + e.getMessage());
            return;
        }

        new Thread(this).start();
    }

    @Override
    public void run() {
        try {
            String header = requestInput.readLine();

            if(header != null) {
                System.out.println("\nSocket connection with " + connection.getRemoteSocketAddress().toString() + " at " + new Date());

                StringTokenizer tokenizer = new StringTokenizer(header);
                String httpMethod = tokenizer.nextToken().toUpperCase();

                switch (httpMethod) {
                    case "GET":
                        GETHandler.handleConnection(responseOutput,responseData,tokenizer);
                        break;
                    case "POST":
                        POSTHandler.handleConnection(requestInput,responseOutput,responseData,tokenizer);
                        break;
                    case "PUT":
                        PUTHandler.handleConnection(requestInput,responseOutput,responseData,tokenizer);
                        break;
                    case "DELETE":
                        DELETEHandler.handleConnection(requestInput,responseOutput,responseData,tokenizer);
                        break;
                    default:
                        String response = "Unsupported or unrecognized HTTP method";
                        ResponseHandler.responseBadRequest(responseOutput,responseData,response);
                        break;
                }
            }
        }
        catch (IOException e) {
            System.err.println("Error processing request: " + e.getMessage());
        }
        finally {
            try {
                requestInput.close();
                responseOutput.close();
                responseData.close();
                connection.close();
            }
            catch (Exception e) {
                System.err.println("Error closing connection: " + e.getMessage());
            }
        }
    }
}
