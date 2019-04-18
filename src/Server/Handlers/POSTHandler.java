package Server.Handlers;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public abstract class POSTHandler {

    public static void handleConnection(BufferedReader requestInput, PrintWriter responseOutput, BufferedOutputStream responseData, StringTokenizer tokenizer) {

        String response = "No POST method for given URL.";
        ResponseHandler.responseNotFound(responseOutput,responseData,response);
    }
}
