package Classes;

public class KeyValuePair {

    public final String key;
    public final String value;
    public boolean used;

    public KeyValuePair(String k, String v) {

        key = k.strip();
        value = v.strip();
        used = false;
    }
}
