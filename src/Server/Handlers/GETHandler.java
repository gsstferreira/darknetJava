package Server.Handlers;

import Server.Methods.GET.GetDetect;

import java.io.BufferedOutputStream;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public abstract class GETHandler {

    public static void handleConnection(PrintWriter responseOutput, BufferedOutputStream responseData, StringTokenizer tokenizer) {

        String[] pathAndParams = tokenizer.nextToken().split("\\?");

        switch(pathAndParams[0]) {

            case "/Detect":
                GetDetect.detect(responseOutput,responseData,pathAndParams[1]);
                break;
            default:
                String response = "No GET request for given URL.";
                ResponseHandler.responseNotFound(responseOutput,responseData,response);
                break;
        }
    }
}
