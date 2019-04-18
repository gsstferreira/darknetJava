package Server.Handlers;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Date;

public abstract class ResponseHandler {

    public static void responseOk(PrintWriter responseOutput, BufferedOutputStream responseData, String data, String contentType) {

        try {

            if(data != null && !data.isEmpty()) {
                responseOutput.println("HTTP/1.1 200 OK");
                responseOutput.println("Server: YOLO Server");
                responseOutput.println("Date: " + new Date());
                responseOutput.println("Content-type: " + contentType);
                responseOutput.println("Content-length: " + data.length());
                responseOutput.println();
                responseOutput.flush();

                responseData.write(data.getBytes(), 0, data.length());
                responseData.flush();
            }
            else {
                responseOutput.println("HTTP/1.1 204 NO CONTENT");
                responseOutput.println("Server: YOLO Server");
                responseOutput.println("Date: " + new Date());
                responseOutput.println();
                responseOutput.flush();
            }
        }
        catch (IOException e) {
            System.err.println("Error sending response: " + e.getMessage());
        }
    }

    public static void responseBadRequest(PrintWriter responseOutput, BufferedOutputStream responseData, String data) {

        try {
            responseOutput.println("HTTP/1.1 400 BAD REQUEST");
            responseOutput.println("Server: YOLO Server");
            responseOutput.println("Date: " + new Date());

            responseOutput.println("Content-type: text/html");
            responseOutput.println("Content-length: " + data.length());
            responseOutput.println();
            responseOutput.flush();

            responseData.write(data.getBytes(), 0, data.length());
            responseData.flush();
        }
        catch (IOException e) {
            System.err.println("Error sending response: " + e.getMessage());
        }
    }

    public static void responseNotFound(PrintWriter responseOutput, BufferedOutputStream responseData, String data) {

        try {
            responseOutput.println("HTTP/1.1 404 NOT FOUND");
            responseOutput.println("Server: YOLO Server");
            responseOutput.println("Date: " + new Date());

            responseOutput.println("Content-type: text/html");
            responseOutput.println("Content-length: " + data.length());
            responseOutput.println();
            responseOutput.flush();

            responseData.write(data.getBytes(), 0, data.length());
            responseData.flush();
        }
        catch (IOException e) {
            System.err.println("Error sending response: " + e.getMessage());
        }
    }

    public static void responseInternalServerError(PrintWriter responseOutput, BufferedOutputStream responseData, String data) {

        try {
            responseOutput.println("HTTP/1.1 500 INTERNAL SERVER ERROR");
            responseOutput.println("Server: YOLO Server");
            responseOutput.println("Date: " + new Date());

            responseOutput.println("Content-type: text/html");
            responseOutput.println("Content-length: " + data.length());
            responseOutput.println();
            responseOutput.flush();

            responseData.write(data.getBytes(), 0, data.length());
            responseData.flush();
        }
        catch (IOException e) {
            System.err.println("Error sending response: " + e.getMessage());
        }
    }
}
