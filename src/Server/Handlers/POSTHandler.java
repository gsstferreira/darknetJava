package Server.Handlers;

import Server.Methods.POST.PostDetect;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.StringTokenizer;

abstract class POSTHandler {

    static void handleConnection(BufferedReader requestInput, PrintWriter responseOutput, BufferedOutputStream responseData, StringTokenizer tokenizer) {

        String[] pathAndParams = tokenizer.nextToken().split("\\?");

        StringBuilder sb = new StringBuilder();

        //noinspection SwitchStatementWithTooFewBranches
        switch (pathAndParams[0]){
            case "/Detect":
                try {
                    //noinspection StatementWithEmptyBody
                    while((requestInput.readLine()).length() != 0) {
                        //System.out.println(s.replace("\n","").replace("\r",""));
                    }
                    while (requestInput.ready()) {
                        sb.append((char)requestInput.read());
                    }
                    PostDetect.detect(responseOutput,responseData,sb.toString());
                }
                catch (IOException e) {
                    String response = "Error trying to read request payload";
                    ResponseHandler.responseInternalServerError(responseOutput,responseData,response);
                }
                break;
            default:
                String response = "No POST method for given URL.";
                ResponseHandler.responseNotFound(responseOutput,responseData,response);
                break;
        }
    }
}
