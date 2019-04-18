package Server.Handlers;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Date;
import java.util.StringTokenizer;

public abstract class DELETEHandler {

    public static void handleConnection(BufferedReader requestInput, PrintWriter responseOutput, BufferedOutputStream responseData, StringTokenizer tokenizer) {

        String response = "No DELETE method for given URL.";
        ResponseHandler.responseNotFound(responseOutput,responseData,response);
    }
}
